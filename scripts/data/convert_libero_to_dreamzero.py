"""Convert the openpi-style LIBERO LeRobot dataset (physical-intelligence/libero) into the
DreamZero / GEAR LeRobot format expected by DreamZero's sharded video dataloader.

Why this is needed
------------------
`physical-intelligence/libero` stores camera frames *inside* the parquet files as encoded
PNG bytes (``info.json`` features ``image`` / ``wrist_image`` have ``dtype: "image"`` and
``total_videos: 0``). DreamZero's loader
(``ShardedLeRobotSubLangSingleActionChunkDatasetDROID``) instead reads frames from on-disk
MP4 videos under ``videos/`` via ``get_frames_by_timestamps`` (decord). It also needs a
``meta/modality.json`` mapping short modality keys -> parquet/video columns, and per-column
normalization stats (with q01/q99) in ``meta/stats.json``.

This script:
  1. Re-encodes ``image`` / ``wrist_image`` to MP4 (one file per view per episode) at the
     dataset fps, under ``videos/chunk-XXX/observation.images.<name>/episode_*.mp4``.
  2. Rewrites each parquet WITHOUT the image columns (keeping state, actions, timestamp,
     frame_index, episode_index, index, task_index).
  3. Writes ``meta/info.json`` with the two cameras as ``dtype: "video"`` features.
  4. Writes ``meta/modality.json`` for the ``libero_sim`` embodiment
     (state: eef_position[0:3], eef_rotation[3:6], gripper_state[6:8];
      action: eef_delta[0:6], gripper_action[6:7]; 2 video views; task annotation).
  5. Computes ``meta/stats.json`` with mean/std/min/max/q01/q99 for ``state`` and ``actions``.
  6. Writes ``meta/embodiment.json`` and copies ``episodes.jsonl`` / ``tasks.jsonl``.

Usage:
    python scripts/data/convert_libero_to_dreamzero.py \
        --src data/libero_raw_lerobot \
        --dst data/libero_lerobot \
        --num-workers 64
"""

import argparse
import io
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image

VIDEO_FEATURE_NAMES = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
}
# parquet columns kept in the rewritten data files (everything except the image columns)
KEEP_COLUMNS = [
    "state",
    "actions",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]


def _decode_image_cell(cell) -> np.ndarray:
    """Decode a LeRobot v2.0 image cell ({'bytes': png, 'path': ...}) to HxWxC uint8 RGB."""
    if isinstance(cell, dict):
        data = cell.get("bytes")
    elif isinstance(cell, (bytes, bytearray)):
        data = bytes(cell)
    else:
        # already an array
        arr = np.asarray(cell)
        return arr.astype(np.uint8)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _write_video(frames: np.ndarray, path: str, fps: float) -> int:
    """Write frames (T,H,W,C uint8) as a constant-frame-rate H.264 MP4. Returns frames written."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=9,
        macro_block_size=1,  # do not silently resize to multiples of 16
        ffmpeg_params=["-pix_fmt", "yuv420p", "-g", "1"],  # -g 1: all keyframes -> exact seek
    )
    n = 0
    for f in frames:
        writer.append_data(f)
        n += 1
    writer.close()
    return n


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def convert_one_episode(args):
    (src, dst, episode_index, chunks_size, fps) = args
    chunk = _episode_chunk(episode_index, chunks_size)
    src_parquet = os.path.join(
        src, f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    df = pd.read_parquet(src_parquet)

    # Decode + write videos
    for short_name, feature_name in VIDEO_FEATURE_NAMES.items():
        frames = np.stack([_decode_image_cell(c) for c in df[short_name].tolist()], axis=0)
        out_video = os.path.join(
            dst,
            f"videos/chunk-{chunk:03d}/{feature_name}/episode_{episode_index:06d}.mp4",
        )
        nwritten = _write_video(frames, out_video, fps)
        if nwritten != len(df):
            raise RuntimeError(
                f"episode {episode_index} {feature_name}: wrote {nwritten} frames != {len(df)} rows"
            )

    # Rewrite parquet without image columns
    keep = [c for c in KEEP_COLUMNS if c in df.columns]
    out_df = df[keep].copy()
    out_parquet = os.path.join(
        dst, f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)

    # Return state/action for global stats
    state = np.stack(df["state"].tolist(), axis=0).astype(np.float64)
    actions = np.stack(df["actions"].tolist(), axis=0).astype(np.float64)
    return episode_index, state, actions, len(df)


def compute_stats(all_state: np.ndarray, all_actions: np.ndarray) -> dict:
    def col_stats(x):
        return {
            "mean": np.mean(x, axis=0).tolist(),
            "std": (np.std(x, axis=0) + 1e-8).tolist(),
            "min": np.min(x, axis=0).tolist(),
            "max": np.max(x, axis=0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(),
        }

    return {"state": col_stats(all_state), "actions": col_stats(all_actions)}


def build_modality_json() -> dict:
    return {
        "state": {
            "eef_position": {"original_key": "state", "start": 0, "end": 3,
                             "rotation_type": None, "absolute": True, "dtype": "float32"},
            "eef_rotation": {"original_key": "state", "start": 3, "end": 6,
                             "rotation_type": None, "absolute": True, "dtype": "float32"},
            "gripper_state": {"original_key": "state", "start": 6, "end": 8,
                              "rotation_type": None, "absolute": True, "dtype": "float32"},
        },
        "action": {
            "eef_delta": {"original_key": "actions", "start": 0, "end": 6,
                          "rotation_type": None, "absolute": False, "dtype": "float32"},
            "gripper_action": {"original_key": "actions", "start": 6, "end": 7,
                               "rotation_type": None, "absolute": False, "dtype": "float32"},
        },
        "video": {
            "image": {"original_key": "observation.images.image"},
            "wrist_image": {"original_key": "observation.images.wrist_image"},
        },
        "annotation": {
            "task": {"original_key": "task_index"},
        },
    }


def build_info_json(src_info: dict, total_episodes: int, fps: float) -> dict:
    info = json.loads(json.dumps(src_info))  # deep copy
    info["total_videos"] = total_episodes * len(VIDEO_FEATURE_NAMES)
    feats = info["features"]
    for short_name, feature_name in VIDEO_FEATURE_NAMES.items():
        old = feats.pop(short_name)
        h = old["shape"][0]
        w = old["shape"][1]
        c = old["shape"][2] if len(old["shape"]) > 2 else 3
        feats[feature_name] = {
            "dtype": "video",
            "shape": [h, w, c],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": float(fps),
                "video.height": int(h),
                "video.width": int(w),
                "video.channels": int(c),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/libero_raw_lerobot")
    ap.add_argument("--dst", default="data/libero_lerobot")
    ap.add_argument("--num-workers", type=int, default=min(64, os.cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=0, help="only convert first N episodes (debug)")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    os.makedirs(os.path.join(dst, "meta"), exist_ok=True)

    src_info = json.load(open(os.path.join(src, "meta/info.json")))
    total_episodes = src_info["total_episodes"]
    chunks_size = src_info["chunks_size"]
    fps = src_info["fps"]
    if args.limit:
        total_episodes = min(total_episodes, args.limit)

    print(f"Converting {total_episodes} episodes (fps={fps}, chunks_size={chunks_size}) "
          f"with {args.num_workers} workers")

    tasks = [(src, dst, ep, chunks_size, fps) for ep in range(total_episodes)]
    all_state_parts = [None] * total_episodes
    all_action_parts = [None] * total_episodes
    done = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(convert_one_episode, t) for t in tasks]
        for fut in as_completed(futures):
            ep, state, actions, n = fut.result()
            all_state_parts[ep] = state
            all_action_parts[ep] = actions
            done += 1
            if done % 100 == 0 or done == total_episodes:
                print(f"  [{done}/{total_episodes}] episodes done", flush=True)

    # Stats
    all_state = np.concatenate(all_state_parts, axis=0)
    all_actions = np.concatenate(all_action_parts, axis=0)
    print(f"Computing stats over {all_state.shape[0]} frames "
          f"(state dim {all_state.shape[1]}, action dim {all_actions.shape[1]})")
    stats = compute_stats(all_state, all_actions)
    json.dump(stats, open(os.path.join(dst, "meta/stats.json"), "w"), indent=2)

    # info.json / modality.json / embodiment.json
    json.dump(build_info_json(src_info, total_episodes, fps),
              open(os.path.join(dst, "meta/info.json"), "w"), indent=2)
    json.dump(build_modality_json(),
              open(os.path.join(dst, "meta/modality.json"), "w"), indent=2)
    json.dump({"embodiment_tag": "libero_sim"},
              open(os.path.join(dst, "meta/embodiment.json"), "w"), indent=2)

    # copy tasks.jsonl, episodes.jsonl (truncate episodes if --limit)
    shutil.copy(os.path.join(src, "meta/tasks.jsonl"), os.path.join(dst, "meta/tasks.jsonl"))
    with open(os.path.join(src, "meta/episodes.jsonl")) as f:
        ep_lines = f.readlines()
    if args.limit:
        ep_lines = ep_lines[:total_episodes]
    with open(os.path.join(dst, "meta/episodes.jsonl"), "w") as f:
        f.writelines(ep_lines)

    print(f"Done. Converted dataset at: {dst}")


if __name__ == "__main__":
    main()
