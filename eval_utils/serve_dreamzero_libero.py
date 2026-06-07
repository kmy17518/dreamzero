"""Serve a DreamZero policy for LIBERO simulation evaluation over the websocket policy server.

This is the server half of the LIBERO eval (the LIBERO sim runs in a separate process/env and
talks to this server via eval_utils/policy_client.py). It wraps groot.vla.model.n1_5.sim_policy
.GrootSimPolicy and adapts LIBERO observations -> model inputs and model actions -> the 7-dim
LIBERO action (6 delta-eef + 1 gripper).

Two embodiments are supported via --embodiment_tag:

  * libero_sim  : a DreamZero model trained on the LIBERO embodiment (2 views, 8-dim eef state,
                  7-dim delta action). This is the real LIBERO policy path.
  * oxe_droid   : the public DreamZero-DROID checkpoint (3 views, joint-position action). This is
                  ONLY a plumbing/harness test: LIBERO observations are adapted to the DROID input
                  format (agentview duplicated into the two exterior views, 8-dim state mapped into
                  joint(7)+gripper(1)). The produced actions are NOT meaningful for LIBERO (DROID is
                  a different embodiment/action space) but let us verify the full eval loop runs and
                  saves videos/metrics before a LIBERO model is trained.

Usage (single GPU):
  python eval_utils/serve_dreamzero_libero.py \
      --model_path ./checkpoints/DreamZero-DROID --embodiment_tag oxe_droid --port 8000

  python eval_utils/serve_dreamzero_libero.py \
      --model_path ./checkpoints/dreamzero_libero_wan22 --embodiment_tag libero_sim --port 8000

The LIBERO client (eval_utils/run_libero_eval.py) sends, per step:
  observation/image       (H,W,3) uint8  -- agentview, already rotated 180 to match training
  observation/wrist_image (H,W,3) uint8  -- wrist, already rotated 180
  observation/state       (8,)   float   -- [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
  prompt                  str
  session_id              str            -- episode boundary (resets KV cache on change)
Response: {"actions": (N, 7)} float32.
"""

import copy
import datetime
import logging
import os
import sys

import imageio

logger = logging.getLogger(__name__)

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import tyro

_dynamo = torch._dynamo.config
if hasattr(_dynamo, "cache_size_limit"):
    _dynamo.cache_size_limit = 1000
if hasattr(_dynamo, "recompile_limit"):
    _dynamo.recompile_limit = 800
if hasattr(_dynamo, "accumulated_cache_size_limit"):
    _dynamo.accumulated_cache_size_limit = 1000
if hasattr(_dynamo, "accumulated_recompile_limit"):
    _dynamo.accumulated_recompile_limit = 2000
from pathlib import Path
from tianshou.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpi_client.base_policy import BasePolicy

from eval_utils.policy_server import WebsocketPolicyServer, PolicyServerConfig
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
from groot.vla.data.transform import ComposedModalityTransform


DEFAULT_IMAGE_HEIGHT = 160
DEFAULT_IMAGE_WIDTH = 320
FRAMES_PER_CHUNK = 4  # frames sent after the first call; the causal action head repeats these
                      # internally to num_frame_per_block (works for num_frame_per_block in {2,4}).

# Per-embodiment adapter: how to map LIBERO obs -> model keys, and model action -> 7-dim LIBERO action.
#   video_fill : list of (model_video_key, source) where source in {"agent", "wrist"}
#   state_split: list of (model_state_key, start, end) slicing the 8-dim LIBERO state vector
#   language_key: model annotation key to put the prompt under
#   action_keys: model action.* keys to concatenate (in order) into the raw action vector
EMB_CONFIG = {
    "libero_sim": {
        "video_fill": [
            ("video.image", "agent"),
            ("video.wrist_image", "wrist"),
        ],
        "state_split": [
            ("state.eef_position", 0, 3),
            ("state.eef_rotation", 3, 6),
            ("state.gripper_state", 6, 8),
        ],
        "language_key": "annotation.task",
        "action_keys": ["action.eef_delta", "action.gripper_action"],
    },
    "oxe_droid": {
        "video_fill": [
            ("video.exterior_image_1_left", "agent"),
            ("video.exterior_image_2_left", "agent"),  # duplicate agentview (DROID expects 3 views)
            ("video.wrist_image_left", "wrist"),
        ],
        "state_split": [
            ("state.joint_position", 0, 7),
            ("state.gripper_position", 7, 8),
        ],
        "language_key": "annotation.language.action_text",
        "action_keys": ["action.joint_position", "action.gripper_position"],
    },
}


def _get_expected_video_resolution(policy: GrootSimPolicy) -> tuple[int, int]:
    eval_transform = getattr(policy, "eval_transform", None)
    if eval_transform is None or not isinstance(eval_transform, ComposedModalityTransform):
        return (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
    for t in eval_transform.transforms:
        if hasattr(t, "original_resolutions") and getattr(t, "original_resolutions", None):
            res = t.original_resolutions
            if res:
                w, h = next(iter(res.values()))  # stored as (width, height)
                return (int(h), int(w))
    return (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)


def _resize_frames(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if frames.ndim == 3:
        if (frames.shape[0], frames.shape[1]) != (target_h, target_w):
            frames = cv2.resize(frames, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return frames
    return np.stack(
        [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for f in frames],
        axis=0,
    )


def _maybe_init_distributed():
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


class DreamZeroLiberoPolicy(BasePolicy):
    """Wraps GrootSimPolicy for LIBERO sim eval. Converts LIBERO obs <-> model batch."""

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        image_height: int,
        image_width: int,
        embodiment_tag: str = "libero_sim",
        save_video_pred: bool = False,
        video_output_dir: str = "./video_pred_output",
        future_frame_shift: bool | None = None,
    ):
        super().__init__()
        self._policy = groot_policy
        self._image_height = image_height
        self._image_width = image_width
        # Explicit-conditioning future-frame shift: the model predicts video one block ahead of the
        # action it co-produces, so the first generated block carries the unsupervised a_{-1} chunk.
        # We prime once per episode (return a no-op for the first query) so the KV cache advances and
        # every subsequent query returns a correctly-aligned action (query N -> a_{N-2}; the robot is
        # at window N-2 after the one-query idle). Auto-detected from the trained model's config.
        detected_ffs = False
        try:
            detected_ffs = bool(
                getattr(groot_policy.trained_model.action_head.config, "future_frame_shift", False)
            )
        except Exception:  # noqa: BLE001 - config layout may vary; default to off
            detected_ffs = False
        self._future_frame_shift = detected_ffs if future_frame_shift is None else future_frame_shift
        if self._future_frame_shift:
            logger.info(
                "future_frame_shift ON: priming first query of each episode with a no-op chunk "
                "(discards the unsupervised a_{-1}); later queries are correctly aligned."
            )
        if embodiment_tag not in EMB_CONFIG:
            raise ValueError(
                f"Unsupported embodiment_tag={embodiment_tag}; expected one of {list(EMB_CONFIG)}"
            )
        self._embodiment_tag = embodiment_tag
        self._cfg = EMB_CONFIG[embodiment_tag]
        self._video_keys = [mk for (mk, _src) in self._cfg["video_fill"]]
        self._frame_buffers = {k: [] for k in self._video_keys}
        self._is_first_call = True
        self._current_session_id = None
        self._save_video_pred = save_video_pred
        self._video_output_dir = video_output_dir
        self._video_pred_latents: list[torch.Tensor] = []
        self._current_prompt: str = ""

    def _convert_observation(self, obs: dict) -> dict:
        agent = obs.get("observation/image")
        wrist = obs.get("observation/wrist_image")
        sources = {"agent": agent, "wrist": wrist}

        for model_key, src in self._cfg["video_fill"]:
            data = sources[src]
            if data is None:
                continue
            data = np.asarray(data)
            data = _resize_frames(data, self._image_height, self._image_width)
            if data.ndim == 4:
                self._frame_buffers[model_key].extend(list(data))
            else:
                self._frame_buffers[model_key].append(data)

        num_frames = 1 if self._is_first_call else FRAMES_PER_CHUNK
        converted = {}
        for model_key, buffer in self._frame_buffers.items():
            if len(buffer) == 0:
                continue
            if len(buffer) >= num_frames:
                frames_to_use = buffer[-num_frames:]
            else:
                frames_to_use = buffer.copy()
                while len(frames_to_use) < num_frames:
                    frames_to_use.insert(0, buffer[0])
            converted[model_key] = np.stack(frames_to_use, axis=0)

        # State: slice the 8-dim LIBERO state vector into the model's state sub-keys.
        state_vec = np.asarray(obs.get("observation/state", np.zeros(8, dtype=np.float64)))
        state_vec = state_vec.reshape(-1).astype(np.float64)
        for model_key, start, end in self._cfg["state_split"]:
            seg = state_vec[start:end]
            if seg.shape[0] < (end - start):  # pad if client sent a shorter state
                seg = np.pad(seg, (0, (end - start) - seg.shape[0]))
            converted[model_key] = seg.reshape(1, -1)

        prompt = obs.get("prompt", "")
        if prompt:
            self._current_prompt = prompt
        converted[self._cfg["language_key"]] = prompt
        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Assemble action.* keys in the configured order, then return 7-dim [move(6), gripper(1)]."""
        parts = []
        for key in self._cfg["action_keys"]:
            if key not in action_dict or action_dict[key] is None:
                continue
            val = action_dict[key]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            val = np.asarray(val)
            if val.ndim == 1:
                val = val.reshape(-1, 1)
            parts.append(val.astype(np.float32))
        if not parts:
            return np.zeros((1, 7), dtype=np.float32)
        full = np.concatenate(parts, axis=-1)  # (N, D) ; D=7 for libero, 8 for droid
        move = full[:, :6]
        gripper = full[:, -1:]
        return np.concatenate([move, gripper], axis=-1).astype(np.float32)  # (N, 7)

    def infer(self, obs: dict) -> dict:
        session_id = obs.get("session_id")
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                self.reset({})
            self._current_session_id = session_id

        converted_obs = self._convert_observation(obs)
        batch = Batch(obs=converted_obs)
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        if self._save_video_pred and video_pred is not None:
            self._video_pred_latents.append(video_pred.detach())

        action_chunk = result_batch.act
        action_dict = {k: v for k, v in action_chunk.items() if k.startswith("action.")}
        action = self._convert_action(action_dict)

        # Future-frame shift priming. The shifted model emits the action one block behind the video,
        # so the FIRST generated block of every autoregressive sequence carries the unsupervised
        # a_{-1} chunk. A fresh sequence happens at episode start AND every time the action head
        # resets its KV cache (current_start_frame >= local_attn_size, ~every max_chunk_size queries).
        # Detect it via current_start_frame == 1 + num_frame_per_block (warmup +1, then one block):
        #   - episode start (single obs frame): idle one query (no-op); the next query returns a_0.
        #   - mid-episode reset (>=FRAMES_PER_CHUNK obs frames): regenerate once on the same obs to
        #     fetch the aligned a_0 (avoids a recurring ~replan_steps idle at every cache reset).
        if self._future_frame_shift:
            ah = getattr(self._policy.trained_model, "action_head", None)
            nfpb = int(getattr(ah, "num_frame_per_block", 2)) if ah is not None else 2
            cf = int(getattr(ah, "current_start_frame", -1)) if ah is not None else -1
            if cf == 1 + nfpb:  # just generated a fresh sequence's first block -> a_{-1}
                if self._is_first_call:
                    logger.info("[future_frame_shift] episode start: returning no-op priming chunk (discarded a_{-1}).")
                    action = np.zeros_like(action)
                else:
                    logger.info("[future_frame_shift] cache reset (cf=%d): regenerating aligned chunk a_0.", cf)
                    batch2 = Batch(obs=copy.deepcopy(converted_obs))
                    with torch.no_grad():
                        result_batch2, video_pred2 = self._policy.lazy_joint_forward_causal(batch2)
                    if self._save_video_pred and video_pred2 is not None:
                        self._video_pred_latents.append(video_pred2.detach())
                    action_dict2 = {k: v for k, v in result_batch2.act.items() if k.startswith("action.")}
                    action = self._convert_action(action_dict2)

        if self._is_first_call:
            self._is_first_call = False
        return {"actions": action}

    def _save_predicted_video(self) -> None:
        if not self._video_pred_latents:
            return
        try:
            from einops import rearrange

            action_head = self._policy.trained_model.action_head
            latents = torch.cat(self._video_pred_latents, dim=2)
            with torch.no_grad():
                frames = action_head.vae.decode(
                    latents,
                    tiled=action_head.tiled,
                    tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                    tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
                )
            frames = rearrange(frames, "B C T H W -> B T H W C")[0]
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            os.makedirs(self._video_output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
            existing = [f for f in os.listdir(self._video_output_dir) if f.endswith(".mp4")]
            safe_prompt = "".join(c for c in self._current_prompt.replace(" ", "_") if c.isalnum() or c in "_-.")[:80] or "no_prompt"
            out = os.path.join(self._video_output_dir, f"{len(existing):06}_{safe_prompt}_{timestamp}.mp4")
            imageio.mimsave(out, list(frames), fps=5, codec="libx264")
            logger.info("Saved video prediction (%d frames) to %s", len(frames), out)
        except Exception as e:
            logger.warning("Failed to save video prediction: %s", e)

    def reset(self, reset_info: dict) -> None:
        if self._save_video_pred:
            self._save_predicted_video()
        self._video_pred_latents.clear()
        self._current_prompt = ""
        for key in self._frame_buffers:
            self._frame_buffers[key] = []
        self._is_first_call = True
        self._current_session_id = None
        ah = getattr(self._policy.trained_model, "action_head", None)
        if ah is not None and hasattr(ah, "current_start_frame"):
            ah.current_start_frame = 0
        if ah is not None and hasattr(ah, "language"):
            ah.language = None


def main(
    model_path: str = "./checkpoints/dreamzero_libero_wan22",
    embodiment_tag: str = "libero_sim",
    tokenizer_path: str | None = None,
    port: int = 8000,
    host: str = "0.0.0.0",
    image_height: int | None = None,
    image_width: int | None = None,
    save_video_pred: bool = False,
    video_output_dir: str = "./video_pred_output",
    model_config_overrides: list[str] | None = None,
    future_frame_shift: bool | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    _maybe_init_distributed()
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    logger.info("Loading DreamZero LIBERO policy from %s (embodiment=%s)", model_path, embodiment_tag)
    checkpoint_name = os.path.basename(model_path.rstrip("/"))
    video_output_dir = os.path.join(video_output_dir, checkpoint_name)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        model_config_overrides=list(model_config_overrides) if model_config_overrides else [],
        tokenizer_path_override=tokenizer_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )
    if image_height is not None and image_width is not None:
        h, w = image_height, image_width
        logger.info("Using CLI video resolution: %dx%d", h, w)
    else:
        h, w = _get_expected_video_resolution(policy)
        logger.info("Using checkpoint video resolution: %dx%d (HxW)", h, w)

    wrapper = DreamZeroLiberoPolicy(
        groot_policy=policy,
        image_height=h,
        image_width=w,
        embodiment_tag=embodiment_tag,
        save_video_pred=save_video_pred,
        video_output_dir=video_output_dir,
        future_frame_shift=future_frame_shift,
    )

    server_config = PolicyServerConfig(
        image_resolution=(h, w),
        needs_wrist_camera=True,
        n_external_cameras=1,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="cartesian_position",
    )
    logger.info("Starting WebsocketPolicyServer on %s:%d (DreamZero LIBERO, %dx%d)", host, port, h, w)
    server = WebsocketPolicyServer(policy=wrapper, server_config=server_config, host=host, port=port)
    server.serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
