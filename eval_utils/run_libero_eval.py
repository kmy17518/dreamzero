"""LIBERO simulation evaluation client for DreamZero.

This is the client half of the LIBERO eval. It runs the LIBERO MuJoCo simulation (in the
`dreamzero_libero` conda env) and queries a DreamZero policy server
(eval_utils/serve_dreamzero_libero.py, running in the `dreamzero` env on a GPU) over websocket.

It mirrors openpi/examples/libero/main.py's rollout logic (same task suites, same image
preprocessing incl. the 180-degree rotation, same replanning, same success bookkeeping) but:
  * speaks DreamZero's websocket protocol via eval_utils/policy_client.WebsocketClientPolicy
    (sends an `endpoint` field, a per-episode `session_id`, and calls reset()), and
  * additionally writes a metrics JSON (per-task and overall success rates) next to the videos.

Usage (in the dreamzero_libero env):
  MUJOCO_GL=egl python eval_utils/run_libero_eval.py \
      --host 0.0.0.0 --port 8000 \
      --task-suite-name libero_spatial \
      --num-trials-per-task 5 \
      --video-out-path ./eval_outputs/libero_spatial
"""

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys
import time

import imageio
import numpy as np
import tqdm
import tyro

# DreamZero websocket client (sends `endpoint` field; compatible with eval_utils/policy_server.py).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from eval_utils.policy_client import WebsocketClientPolicy

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 256  # frames are sent at this square resolution; the server resizes to the model res
    replan_steps: int = 5

    # LIBERO
    task_suite_name: str = "libero_spatial"  # libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    max_tasks: int = 0  # 0 = all tasks in the suite; >0 limits tasks (for smoke tests)
    max_steps_override: int = 0  # 0 = use the suite default; >0 overrides the per-episode step cap

    # Outputs
    video_out_path: str = "./eval_outputs/libero/videos"
    metrics_out_path: str = ""  # default: <video_out_path>/../metrics.json
    save_videos: bool = True

    seed: int = 7


def eval_libero(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} ({num_tasks_in_suite} tasks)")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_out_path or os.path.join(
        os.path.dirname(os.path.normpath(args.video_out_path)), "metrics.json"
    )

    suite_max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if args.task_suite_name not in suite_max_steps:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = args.max_steps_override or suite_max_steps[args.task_suite_name]

    client = WebsocketClientPolicy(args.host, args.port)
    logging.info("Connected. Server metadata: %s", client.get_server_metadata())

    n_tasks = num_tasks_in_suite if args.max_tasks <= 0 else min(args.max_tasks, num_tasks_in_suite)

    total_episodes, total_successes = 0, 0
    per_task_metrics = []
    t_start = time.time()

    for task_id in tqdm.tqdm(range(n_tasks), desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"task{task_id}", leave=False):
            # New episode -> new session so the server resets its KV cache / start frame.
            session_id = f"{args.task_suite_name}/task{task_id}/ep{episode_idx}"
            try:
                client.reset({"session_id": session_id})
            except Exception as e:  # noqa: BLE001
                logging.warning("reset failed (continuing): %s", e)

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # IMPORTANT: rotate 180 degrees to match training preprocessing.
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = _resize_square(img, args.resize_size)
                    wrist_img = _resize_square(wrist_img, args.resize_size)
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ).astype(np.float32),
                            "prompt": str(task_description),
                            "session_id": session_id,
                        }
                        result = client.infer(element)
                        action_chunk = result["actions"] if isinstance(result, dict) else result
                        action_chunk = np.asarray(action_chunk)
                        assert len(action_chunk) >= args.replan_steps, (
                            f"replan_steps={args.replan_steps} but policy returned {len(action_chunk)} steps"
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(np.asarray(action).tolist())
                    if done:
                        break
                    t += 1
                except Exception as e:  # noqa: BLE001
                    logging.error(f"Caught exception during rollout: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            if done:
                task_successes += 1
                total_successes += 1

            if args.save_videos:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")[:60]
                out = pathlib.Path(args.video_out_path) / f"task{task_id}_ep{episode_idx}_{task_segment}_{suffix}.mp4"
                imageio.mimwrite(out, [np.asarray(x) for x in replay_images], fps=10)

            logging.info(
                "[task %d ep %d] success=%s | task SR %d/%d | total SR %d/%d (%.1f%%)",
                task_id, episode_idx, bool(done), task_successes, task_episodes,
                total_successes, total_episodes, 100.0 * total_successes / max(total_episodes, 1),
            )

        env.close()
        task_sr = float(task_successes) / float(max(task_episodes, 1))
        per_task_metrics.append({
            "task_id": task_id,
            "task_description": task_description,
            "episodes": task_episodes,
            "successes": task_successes,
            "success_rate": task_sr,
        })
        # Persist metrics incrementally so a crash mid-run still leaves partial results.
        _write_metrics(metrics_path, args, per_task_metrics, total_episodes, total_successes, t_start)
        logging.info("Task %d success rate: %.3f", task_id, task_sr)

    overall = float(total_successes) / float(max(total_episodes, 1))
    logging.info("==== DONE: overall success rate %.3f (%d/%d) ====", overall, total_successes, total_episodes)
    _write_metrics(metrics_path, args, per_task_metrics, total_episodes, total_successes, t_start)
    logging.info("Metrics written to %s", metrics_path)


def _write_metrics(metrics_path, args, per_task_metrics, total_episodes, total_successes, t_start):
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    payload = {
        "task_suite_name": args.task_suite_name,
        "num_trials_per_task": args.num_trials_per_task,
        "replan_steps": args.replan_steps,
        "seed": args.seed,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": float(total_successes) / float(max(total_episodes, 1)),
        "elapsed_sec": time.time() - t_start,
        "per_task": per_task_metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)


def _resize_square(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return np.ascontiguousarray(img.astype(np.uint8))
    import cv2
    return np.ascontiguousarray(
        cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.uint8)
    )


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite transform_utils."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    eval_libero(tyro.cli(Args))
