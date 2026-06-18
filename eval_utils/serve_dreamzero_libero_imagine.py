"""Instrumented LIBERO policy server for the *imagined-vs-real* qualitative figure.

This is a thin wrapper around ``eval_utils/serve_dreamzero_libero.py`` that adds **per-episode dumping
of aligned (real-observation, imagined-future-frame) pairs** for the feedback variant
(``--context_mode C``). It changes **no** baseline behavior: it subclasses ``DreamZeroLiberoPolicy``
and only *additionally* records, per query ``t``:

  * ``real_anchors[t]``  -- the current real observation O_t the model conditioned on (model-res RGB,
                            i.e. the last frame of the agent-view buffer used for this query), and
  * ``imagined[t]``      -- the model's generated/imagined next-frame block Ĝ_t for that query,
                            decoded to pixels via the action-head VAE (we keep the last decoded frame,
                            the furthest-ahead forecast = the frontier that gets fed back).

Because ``context_mode=C`` re-anchors every query and feeds back exactly one generated block, the
realized next observation for the imagined frame Ĝ_t is simply the *next* query's real anchor
O_{t+1}. So a faithful "imagined future frame vs realized next observation" overlay is
``imagined[t]`` vs ``real_anchors[t+1]`` -- both produced here, perfectly aligned, no guesswork.

One ``.npz`` is written per completed episode, named by ``session_id`` (= ``<suite>/task<T>/ep<E>``),
into ``--dump_dir/<checkpoint name>/``. As with ``--save_video_pred``, an episode is flushed on the
*next* ``reset()``, so the final episode of a run is only written if another reset follows (run one
extra throwaway episode, or just ignore the last).

Usage (GPU, ``dreamzero`` env) -- mirrors serve_dreamzero_libero.py + ``--dump_dir``:
  CUDA_VISIBLE_DEVICES=7 python eval_utils/serve_dreamzero_libero_imagine.py \
      --model_path .../checkpoint-40000 --embodiment_tag libero_sim \
      --tokenizer_path ./checkpoints/umt5-xxl --port 8123 \
      --context_mode C --dump_dir ./imagine_dump
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import tyro
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.distributed.device_mesh import init_device_mesh

from eval_utils.policy_server import WebsocketPolicyServer, PolicyServerConfig
from eval_utils.serve_dreamzero_libero import (
    DreamZeroLiberoPolicy,
    _get_expected_video_resolution,
    _maybe_init_distributed,
)
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag

logger = logging.getLogger(__name__)


def _sanitize(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "_-.") else "_" for c in (s or "none"))


class ImagineDumpPolicy(DreamZeroLiberoPolicy):
    """``DreamZeroLiberoPolicy`` + per-episode dump of aligned (real obs, imagined next frame)."""

    def __init__(self, *args, dump_dir: str = "./imagine_dump", **kwargs):
        super().__init__(*args, **kwargs)
        self._dump_dir = dump_dir
        os.makedirs(self._dump_dir, exist_ok=True)
        self._ep_real_agent: list[np.ndarray] = []
        self._ep_real_wrist: list[np.ndarray] = []
        # The dump needs the per-query imagined latents, which the parent only buffers when
        # save_video_pred is on -> force it on regardless of the CLI flag.
        self._save_video_pred = True

    def infer(self, obs: dict) -> dict:
        # Parent does obs-conversion (which extends the agent-view frame buffer) and appends this
        # query's imagined latent to self._video_pred_latents. Run it first, then read back the
        # current real observation it conditioned on = the most recent agent-view frame.
        n_before = len(self._video_pred_latents)
        out = super().infer(obs)
        n_after = len(self._video_pred_latents)
        abuf = self._frame_buffers.get("video.image") or []
        wbuf = self._frame_buffers.get("video.wrist_image") or []
        agent = np.asarray(abuf[-1]).copy() if len(abuf) else None
        wrist = np.asarray(wbuf[-1]).copy() if len(wbuf) else None
        # Keep anchors and imagined latents 1:1. context_mode=C appends exactly one latent per query
        # (1 episode-start no-op, 0 regenerations); guard anyway so a stray double-append can't desync.
        if agent is not None and (n_after - n_before) >= 1:
            self._ep_real_agent.append(agent)
            self._ep_real_wrist.append(wrist if wrist is not None else agent)
        return out

    def _decode_last_frame(self, latents: torch.Tensor) -> np.ndarray:
        """Decode a single query's imagined latent block -> last (furthest-ahead) RGB frame uint8."""
        ah = self._policy.trained_model.action_head
        with torch.no_grad():
            frames = ah.vae.decode(
                latents,
                tiled=ah.tiled,
                tile_size=(ah.tile_size_height, ah.tile_size_width),
                tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
            )
        frames = rearrange(frames, "B C T H W -> B T H W C")[0]
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        return frames  # (T, H, W, 3)

    def _dump_episode(self) -> None:
        latents = list(self._video_pred_latents)
        agents = list(self._ep_real_agent)
        wrists = list(self._ep_real_wrist)
        if not latents or not agents:
            return
        session = self._current_session_id or "unknown_session"
        prompt = self._current_prompt or ""
        try:
            imagined_last = []
            imagined_blocks_T = []
            for lat in latents:
                dec = self._decode_last_frame(lat)  # (T,H,W,3)
                imagined_blocks_T.append(int(dec.shape[0]))
                imagined_last.append(dec[-1])
            n = min(len(imagined_last), len(agents), len(wrists))
            real_agent = np.stack(agents[:n], axis=0)
            real_wrist = np.stack(wrists[:n], axis=0)
            imagined = np.stack(imagined_last[:n], axis=0)
            out = os.path.join(self._dump_dir, _sanitize(session) + ".npz")
            np.savez_compressed(
                out,
                # real_anchors kept as an alias of the agent view for backward compat.
                real_anchors=real_agent,             # (Q, H, W, 3) uint8 -- agent O_t per query
                real_agent=real_agent,               # (Q, H, W, 3) uint8 -- agent view O_t
                real_wrist=real_wrist,               # (Q, H, W, 3) uint8 -- wrist view O_t
                imagined=imagined,                   # (Q, H, W, 3) uint8 -- Ĝ_t composite [agent|wrist]
                session_id=np.array(session),
                prompt=np.array(prompt),
                num_frame_per_block=np.array(int(getattr(
                    self._policy.trained_model.action_head, "num_frame_per_block", 2))),
                imagined_block_pixel_T=np.array(imagined_blocks_T, dtype=np.int64),
            )
            logger.info(
                "[imagine-dump] %s: Q=%d real_agent=%s imagined=%s block_pixelT=%s -> %s",
                session, n, real_agent.shape, imagined.shape, imagined_blocks_T[:1], out,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[imagine-dump] failed for %s: %s", session, e)

    def reset(self, reset_info: dict) -> None:
        # Dump the just-finished episode BEFORE the parent clears latents/prompt/session.
        self._dump_episode()
        super().reset(reset_info)
        self._ep_real_agent = []
        self._ep_real_wrist = []


def main(
    model_path: str = "./checkpoints/dreamzero_libero_wan22",
    embodiment_tag: str = "libero_sim",
    tokenizer_path: str | None = None,
    port: int = 8123,
    host: str = "0.0.0.0",
    image_height: int | None = None,
    image_width: int | None = None,
    save_video_pred: bool = False,
    video_output_dir: str = "./video_pred_output",
    dump_dir: str = "./imagine_dump",
    model_config_overrides: list[str] | None = None,
    future_frame_shift: bool | None = None,
    context_mode: str = "C",
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    _maybe_init_distributed()
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    logger.info("Loading DreamZero LIBERO policy from %s (embodiment=%s)", model_path, embodiment_tag)
    checkpoint_name = os.path.basename(model_path.rstrip("/"))
    video_output_dir = os.path.join(video_output_dir, checkpoint_name)
    dump_dir = os.path.join(dump_dir, checkpoint_name)
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
    else:
        h, w = _get_expected_video_resolution(policy)
    logger.info("Video resolution: %dx%d (HxW)", h, w)

    wrapper = ImagineDumpPolicy(
        groot_policy=policy,
        image_height=h,
        image_width=w,
        embodiment_tag=embodiment_tag,
        save_video_pred=save_video_pred,
        video_output_dir=video_output_dir,
        future_frame_shift=future_frame_shift,
        context_mode=context_mode,
        dump_dir=dump_dir,
    )

    server_config = PolicyServerConfig(
        image_resolution=(h, w),
        needs_wrist_camera=True,
        n_external_cameras=1,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="cartesian_position",
    )
    logger.info("Starting WebsocketPolicyServer on %s:%d (imagine-dump, context_mode=%s, dump_dir=%s)",
                host, port, context_mode, dump_dir)
    server = WebsocketPolicyServer(policy=wrapper, server_config=server_config, host=host, port=port)
    server.serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
