"""
Serve a DreamZero policy for the LIBERO benchmark over the **openpi** websocket protocol.

This server speaks the protocol used by ``openpi``'s LIBERO eval client
(``third_party/openpi/examples/libero/main.py`` -> ``openpi_client.websocket_client_policy``):

  Client -> server (one dict per ``infer`` call):
    - "observation/image":       (H, W, 3) uint8   (agentview, already 180-rotated by client)
    - "observation/wrist_image": (H, W, 3) uint8   (eye-in-hand)
    - "observation/state":       (8,) float         (eef_pos[3] + axisangle[3] + gripper[2])
    - "prompt":                  str                (task description)

  Server -> client:
    - {"actions": (N, 7) float32}   N action steps; the client executes ``replan_steps`` of them.

It wraps ``GrootSimPolicy`` (the same class used for all DreamZero inference). Two embodiments
are supported via ``--embodiment_tag``:

  * ``libero``     -> a DreamZero checkpoint trained on LIBERO (this repo's libero_relative_wan22).
                      Cameras map 1:1 (image, wrist_image); action is 7-dim.
  * ``oxe_droid``  -> the released 14B DreamZero-DROID checkpoint, used only to VERIFY that the
                      eval harness runs end-to-end. LIBERO's 2 cameras are mapped onto DROID's 3
                      views (agentview duplicated for the second exterior view) and the model's
                      8-dim joint+gripper action is truncated to LIBERO's 7-dim action space.

Usage:
  # Serve a LIBERO-trained DreamZero 5B checkpoint:
  python eval_utils/serve_dreamzero_libero.py \
      --model_path ./checkpoints/dreamzero_libero_wan22_full_finetune --embodiment_tag libero --port 8000

  # Verify the harness with the released 14B DROID checkpoint:
  python eval_utils/serve_dreamzero_libero.py \
      --model_path ./checkpoints/DreamZero-DROID --embodiment_tag oxe_droid --port 8000

Then run the LIBERO client (in the openpi libero venv):
  python third_party/openpi/examples/libero/main.py --args.task-suite-name libero_10 --args.port 8000
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import tyro

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("serve_dreamzero_libero")

# Increase dynamo cache limits (flow scheduler recompiles under varying shapes when serving).
_dynamo = torch._dynamo.config
for _attr, _val in [
    ("cache_size_limit", 1000),
    ("recompile_limit", 800),
    ("accumulated_cache_size_limit", 1000),
    ("accumulated_recompile_limit", 2000),
]:
    if hasattr(_dynamo, _attr):
        setattr(_dynamo, _attr, _val)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tianshou.data import Batch  # noqa: E402

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402
from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.data.transform import ComposedModalityTransform  # noqa: E402

# openpi server (vendored in third_party). Falls back to the pip-installed package name.
_OPENPI_SERVING = REPO_ROOT.parent / "cs224r_custom_private" / "third_party" / "openpi" / "src"
if (_OPENPI_SERVING / "openpi" / "serving").exists() and str(_OPENPI_SERVING) not in sys.path:
    sys.path.insert(0, str(_OPENPI_SERVING))
from openpi.serving.websocket_policy_server import WebsocketPolicyServer  # noqa: E402
from openpi_client.base_policy import BasePolicy  # noqa: E402


# Per-embodiment mapping from LIBERO observations to DreamZero model input keys.
#   video keys are filled with the current frame(s); state keys use the (sliced) LIBERO state.
EMBODIMENT_SPEC = {
    "libero": {
        "video": {
            # libero camera -> model video key
            "observation/image": "video.image",
            "observation/wrist_image": "video.wrist_image",
        },
        "state": [("state.state", 0, 8)],   # (model_key, start, end) into the 8-dim libero state
        "language": "annotation.task",
        "action_keys": ["action.action"],   # concatenated in this order -> 7-dim
        "action_dim": 7,
    },
    # Map LIBERO onto the 14B DROID model purely to verify the harness runs.
    "oxe_droid": {
        "video": {
            "observation/image": "video.exterior_image_1_left",
            "observation/image_dup": "video.exterior_image_2_left",  # duplicate agentview
            "observation/wrist_image": "video.wrist_image_left",
        },
        # DROID state is joint_position(7)+gripper(1); we only have an 8-dim libero state, so we
        # feed it directly (values are meaningless for DROID -- this is a plumbing check only).
        "state": [("state.joint_position", 0, 7), ("state.gripper_position", 7, 8)],
        "language": "annotation.language.action_text",
        "action_keys": ["action.joint_position", "action.gripper_position"],
        "action_dim": 7,
    },
}


def _resize(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize (H,W,C) or (T,H,W,C) frames with PIL (no cv2 dependency)."""
    from PIL import Image

    def _one(f):
        if f.shape[0] == h and f.shape[1] == w:
            return f
        return np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR))

    if frames.ndim == 3:
        return _one(frames)
    return np.stack([_one(f) for f in frames], axis=0)


def _expected_resolution(policy: GrootSimPolicy, default=(160, 320)) -> tuple[int, int]:
    et = getattr(policy, "eval_transform", None)
    if isinstance(et, ComposedModalityTransform):
        for t in et.transforms:
            res = getattr(t, "original_resolutions", None)
            if res:
                w, h = next(iter(res.values()))
                return int(h), int(w)
    return default


def _maybe_init_distributed():
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


class DreamZeroLiberoPolicy(BasePolicy):
    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        embodiment_tag: str,
        image_height: int,
        image_width: int,
        num_context_frames: int = 1,
        stateless: bool = True,
    ):
        super().__init__()
        self._policy = groot_policy
        self._spec = EMBODIMENT_SPEC[embodiment_tag]
        self._h = image_height
        self._w = image_width
        self._num_ctx = num_context_frames
        self._stateless = stateless
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._prompt = ""

    def _reset_cache(self):
        self._buffers = {}
        ah = getattr(self._policy.trained_model, "action_head", None)
        if ah is not None and hasattr(ah, "current_start_frame"):
            ah.current_start_frame = 0

    def _build_obs(self, obs: dict) -> dict:
        spec = self._spec
        # Prepare the per-camera frame inputs.
        cam_inputs = dict(obs)
        if "observation/image_dup" in spec["video"] and "observation/image" in obs:
            cam_inputs["observation/image_dup"] = obs["observation/image"]

        converted: dict = {}
        for client_key, model_key in spec["video"].items():
            if client_key not in cam_inputs:
                continue
            frame = np.asarray(cam_inputs[client_key])
            frame = _resize(frame, self._h, self._w).astype(np.uint8)
            buf = self._buffers.setdefault(model_key, [])
            buf.append(frame)
            n = self._num_ctx
            if len(buf) >= n:
                clip = buf[-n:]
            else:
                clip = [buf[0]] * (n - len(buf)) + buf
            converted[model_key] = np.stack(clip, axis=0)  # (T,H,W,C)

        state = np.asarray(obs.get("observation/state", np.zeros(8)), dtype=np.float64).reshape(-1)
        for model_key, s, e in spec["state"]:
            seg = state[s:e]
            if seg.size == 0:
                seg = np.zeros(e - s, dtype=np.float64)
            converted[model_key] = seg.reshape(1, -1)

        self._prompt = str(obs.get("prompt", self._prompt))
        converted[spec["language"]] = self._prompt
        return converted

    def _to_actions(self, act_obj) -> np.ndarray:
        spec = self._spec
        parts = []
        for key in spec["action_keys"]:
            v = getattr(act_obj, key, None)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            v = np.asarray(v)
            if v.ndim == 1:
                v = v.reshape(-1, 1)
            parts.append(v)
        if not parts:
            return np.zeros((1, spec["action_dim"]), dtype=np.float32)
        action = np.concatenate(parts, axis=-1).astype(np.float32)
        return action[:, : spec["action_dim"]]

    def infer(self, obs: dict) -> dict:
        if self._stateless:
            self._reset_cache()
        prompt = str(obs.get("prompt", ""))
        if prompt and prompt != self._prompt and not self._stateless:
            self._reset_cache()
        converted = self._build_obs(obs)
        batch = Batch(obs=converted)
        with torch.no_grad():
            result, _ = self._policy.lazy_joint_forward_causal(batch)
        actions = self._to_actions(result.act)
        return {"actions": actions}

    def reset(self) -> None:
        self._reset_cache()
        self._prompt = ""


def main(
    model_path: str,
    embodiment_tag: str = "libero",
    tokenizer_path: str | None = None,
    port: int = 8000,
    host: str = "0.0.0.0",
    image_height: int | None = None,
    image_width: int | None = None,
    num_context_frames: int = 1,
    stateless: bool = True,
) -> None:
    assert embodiment_tag in EMBODIMENT_SPEC, f"embodiment_tag must be one of {list(EMBODIMENT_SPEC)}"

    _maybe_init_distributed()
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    logger.info("Loading DreamZero policy from %s (embodiment=%s)", model_path, embodiment_tag)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        tokenizer_path_override=tokenizer_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    if image_height is not None and image_width is not None:
        h, w = image_height, image_width
    else:
        h, w = _expected_resolution(policy)
    logger.info("Serving with video resolution %dx%d (HxW)", h, w)

    wrapper = DreamZeroLiberoPolicy(
        groot_policy=policy,
        embodiment_tag=embodiment_tag,
        image_height=h,
        image_width=w,
        num_context_frames=num_context_frames,
        stateless=stateless,
    )

    server = WebsocketPolicyServer(
        policy=wrapper,
        host=host,
        port=port,
        metadata={"policy": "dreamzero", "embodiment": embodiment_tag},
    )
    logger.info("Starting openpi WebsocketPolicyServer on %s:%d", host, port)
    server.serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
