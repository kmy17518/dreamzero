"""
Convert an openpi-style LIBERO LeRobot dataset to the GEAR/DreamZero training format.

The openpi LIBERO LeRobot dataset (e.g. ``physical-intelligence/libero`` or the
``libero-10`` / ``libero-90`` conversions under
``/root/cs224r_custom_private/data/libero100``) stores the camera observations as
**inline PNG bytes** inside the parquet files (``dtype: image``) rather than as MP4
videos. DreamZero's data loader reads frames from MP4 files via decord, exactly like
the DROID dataset. This script bridges the gap by:

  1. Decoding the inline PNG frames for every camera and re-encoding them as one MP4
     per (episode, camera), laid out identically to the DROID GEAR format:
         videos/chunk-000/observation.images.<cam>/episode_000000.mp4
  2. Re-writing the parquet files without the (now redundant) image columns so the
     converted dataset is compact.
  3. Generating the GEAR metadata DreamZero needs:
         meta/info.json            (image features rewritten as video features)
         meta/modality.json        (state / action / video / annotation mapping)
         meta/embodiment.json      (embodiment tag, default "libero")
         meta/stats.json           (per-feature statistics for normalization)
         meta/tasks.jsonl          (copied / rebuilt)
         meta/episodes.jsonl       (copied / rebuilt)

LIBERO layout (openpi):
  - columns:   image (PNG), wrist_image (PNG), state (8), actions (7), task_index, ...
  - cameras:   image  -> observation.images.image   (agentview, exterior)
               wrist_image -> observation.images.wrist_image (eye-in-hand)
  - state:     8-dim   = eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
  - actions:   7-dim   = delta eef pose(6) + gripper(1)   (absolute command -> NOT relative)

Usage:
  python scripts/data/convert_libero_to_gear.py \
      --src /root/cs224r_custom_private/data/libero100/libero-10 \
      --out ./data/libero10_gear \
      --embodiment-tag libero

  # In-place metadata + side-car videos (writes videos/ and meta/ next to data/):
  python scripts/data/convert_libero_to_gear.py --src <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Camera column -> GEAR original_key (directory name under videos/).
DEFAULT_CAMERA_MAP = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
}
STATE_COL = "state"
ACTION_COL = "actions"
TASK_COL = "task_index"
# Columns that should be carried over verbatim into the rewritten parquet.
NON_IMAGE_KEEP_COLS = [
    "state", "actions", "timestamp", "frame_index", "episode_index", "index", "task_index",
]


def _decode_png(cell) -> np.ndarray:
    """Decode a LeRobot image cell ({'bytes': ..., 'path': ...} or raw bytes) to HxWx3 uint8."""
    if isinstance(cell, dict):
        data = cell.get("bytes")
        if data is None and cell.get("path"):
            with open(cell["path"], "rb") as f:
                data = f.read()
    else:
        data = cell
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _episode_parquet_path(root: Path, pattern: str, chunk: int, ep: int) -> Path:
    return root / pattern.format(episode_chunk=chunk, episode_index=ep)


def _convert_one_episode(args) -> tuple[int, int, dict]:
    """Worker: decode PNGs -> MP4(s) and write the slimmed parquet for one episode.

    Returns (episode_index, length, {column: stacked float array}) for stats accumulation.
    """
    (src_pq, out_pq, video_targets, fps, camera_map, keep_cols) = args
    df = pd.read_parquet(src_pq)
    length = len(df)

    # 1. Write one MP4 per camera.
    for cam_col, out_mp4 in video_targets.items():
        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        frames = [_decode_png(df.iloc[i][cam_col]) for i in range(length)]
        # libx264 + yuv420p so decord/torchcodec can decode; CFR at `fps` so per-frame
        # timestamps are i/fps and align with the parquet `timestamp` column.
        writer = imageio.get_writer(
            out_mp4.as_posix(),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            pixelformat="yuv420p",
        )
        try:
            for fr in frames:
                writer.append_data(fr)
        finally:
            writer.close()

    # 2. Write slimmed parquet (drop image columns).
    out_pq = Path(out_pq)
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in keep_cols if c in df.columns]
    if out_pq.resolve() != Path(src_pq).resolve():
        df[cols].to_parquet(out_pq, index=False)

    # 3. Accumulate numeric data for stats.
    numeric = {}
    for col in (STATE_COL, ACTION_COL):
        if col in df.columns:
            arr = np.stack(df[col].values).astype(np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            numeric[col] = arr
    ep_idx = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else -1
    return ep_idx, length, numeric


def _compute_stats(all_numeric: dict[str, list[np.ndarray]]) -> dict:
    stats = {}
    for col, chunks in all_numeric.items():
        if not chunks:
            continue
        data = np.concatenate(chunks, axis=0)
        stats[col] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
    return stats


def _build_info(src_info: dict, camera_map: dict[str, str], fps: float, total_episodes: int) -> dict:
    info = json.loads(json.dumps(src_info))  # deep copy
    info["fps"] = fps
    features = info.get("features", {})
    h = w = None
    for cam_col in camera_map:
        feat = features.pop(cam_col, None)
        if feat is not None and feat.get("shape"):
            shp = feat["shape"]
            # image features are stored HxWxC
            h, w = int(shp[0]), int(shp[1])
    if h is None:
        h = w = 256
    for cam_col, orig_key in camera_map.items():
        features[orig_key] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": float(fps),
                "video.channels": 3,
                "video.height": h,
                "video.width": w,
                "video.codec": "h264",
            },
        }
    info["features"] = features
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info["data_path"] = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    info["total_videos"] = total_episodes * len(camera_map)
    return info


def _build_modality(camera_map: dict[str, str], state_dim: int, action_dim: int) -> dict:
    return {
        "state": {
            "state": {
                "original_key": STATE_COL,
                "start": 0,
                "end": state_dim,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            }
        },
        "action": {
            "action": {
                "original_key": ACTION_COL,
                "start": 0,
                "end": action_dim,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            }
        },
        "video": {
            cam_col: {"original_key": orig_key} for cam_col, orig_key in camera_map.items()
        },
        "annotation": {
            "task": {"original_key": TASK_COL},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Path to the openpi LIBERO LeRobot dataset")
    parser.add_argument("--out", required=True, help="Output path for the GEAR dataset")
    parser.add_argument("--embodiment-tag", default="libero")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS (default: from info.json)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Only convert the first N episodes (debug)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output dir")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    inplace = src == out
    if not (src / "meta" / "info.json").exists():
        log.error("No meta/info.json under %s", src)
        sys.exit(1)

    with open(src / "meta" / "info.json") as f:
        src_info = json.load(f)

    fps = float(args.fps) if args.fps is not None else float(src_info.get("fps", 10))
    total_episodes = int(src_info["total_episodes"])
    if args.max_episodes is not None:
        total_episodes = min(total_episodes, args.max_episodes)
    chunks_size = int(src_info.get("chunks_size", 1000))
    data_pattern = src_info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")

    # Determine camera columns actually present.
    feats = src_info.get("features", {})
    camera_map = {c: o for c, o in DEFAULT_CAMERA_MAP.items() if c in feats}
    if not camera_map:
        # Fall back to any image-dtype feature.
        camera_map = {
            c: f"observation.images.{c}" for c, v in feats.items() if v.get("dtype") == "image"
        }
    if not camera_map:
        log.error("No image-dtype camera features found in info.json")
        sys.exit(1)
    log.info("Cameras: %s", camera_map)

    state_dim = int(feats.get(STATE_COL, {}).get("shape", [8])[0])
    action_dim = int(feats.get(ACTION_COL, {}).get("shape", [7])[0])

    if not inplace:
        if out.exists():
            if not args.force:
                log.error("Output %s exists; pass --force to overwrite", out)
                sys.exit(1)
            shutil.rmtree(out)
        (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    # Build per-episode work items.
    work = []
    for ep in range(total_episodes):
        chunk = ep // chunks_size
        src_pq = _episode_parquet_path(src, data_pattern, chunk, ep)
        if not src_pq.exists():
            continue
        out_pq = _episode_parquet_path(out, data_pattern, chunk, ep)
        video_targets = {
            cam_col: (
                out / f"videos/chunk-{chunk:03d}/{orig_key}/episode_{ep:06d}.mp4"
            ).as_posix()
            for cam_col, orig_key in camera_map.items()
        }
        work.append((src_pq.as_posix(), out_pq.as_posix(), video_targets, fps, camera_map, NON_IMAGE_KEEP_COLS))

    log.info("Converting %d episodes (fps=%s) -> %s", len(work), fps, out)
    all_numeric: dict[str, list[np.ndarray]] = {STATE_COL: [], ACTION_COL: []}
    episode_lengths: dict[int, int] = {}

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futs = [ex.submit(_convert_one_episode, w) for w in work]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="episodes"):
                ep_idx, length, numeric = fut.result()
                episode_lengths[ep_idx] = length
                for k, v in numeric.items():
                    all_numeric[k].append(v)
    else:
        for w in tqdm(work, desc="episodes"):
            ep_idx, length, numeric = _convert_one_episode(w)
            episode_lengths[ep_idx] = length
            for k, v in numeric.items():
                all_numeric[k].append(v)

    # ---- Metadata ----
    info = _build_info(src_info, camera_map, fps, len(work))
    info["total_episodes"] = len(work)
    info["total_frames"] = int(sum(episode_lengths.values()))
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    modality = _build_modality(camera_map, state_dim, action_dim)
    with open(out / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)

    embodiment = {"robot_type": args.embodiment_tag, "embodiment_tag": args.embodiment_tag}
    with open(out / "meta" / "embodiment.json", "w") as f:
        json.dump(embodiment, f, indent=4)

    stats = _compute_stats(all_numeric)
    with open(out / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # tasks.jsonl: copy from source if present.
    src_tasks = src / "meta" / "tasks.jsonl"
    if src_tasks.exists():
        shutil.copy(src_tasks, out / "meta" / "tasks.jsonl")

    # episodes.jsonl: copy from source (filter to converted episodes) or rebuild.
    src_episodes = src / "meta" / "episodes.jsonl"
    episodes_out = []
    if src_episodes.exists():
        with open(src_episodes) as f:
            for line in f:
                ep = json.loads(line)
                if ep.get("episode_index", -1) in episode_lengths:
                    episodes_out.append(ep)
    if not episodes_out:
        episodes_out = [
            {"episode_index": ep, "tasks": [""], "length": length}
            for ep, length in sorted(episode_lengths.items())
        ]
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes_out:
            f.write(json.dumps(ep) + "\n")

    print("\n" + "=" * 60)
    print("LIBERO -> GEAR conversion complete!")
    print(f"  Output:          {out}")
    print(f"  Embodiment tag:  {args.embodiment_tag}")
    print(f"  Episodes:        {len(work)}")
    print(f"  Frames:          {info['total_frames']}")
    print(f"  Cameras:         {list(camera_map.values())}")
    print(f"  State dim:       {state_dim}   Action dim: {action_dim}")
    print(f"  FPS:             {fps}")
    print("=" * 60)


if __name__ == "__main__":
    main()
