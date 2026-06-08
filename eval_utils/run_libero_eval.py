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

    # Progress (partial / stage) scoring. These only affect the partial-credit numbers; the final
    # "done" stage of every sub-goal is the exact BDDL predicate, so progress==1.0 <=> env success.
    # Thresholds are heuristics (object body centers have small offsets) -- tune on a few rollouts.
    progress_scores: bool = True
    approach_dist: float = 0.07  # m; gripper<->source-object distance for the "approach source" stage
    place_dist: float = 0.12  # m; source<->target distance for the "approach target" stage

    seed: int = 7


# Predicate families used to decompose a BDDL goal into ordered stages. The whole benchmark uses
# only 6 predicates: on/in are "pick-and-place" (4 stages), open/close/turnon/turnoff are
# "articulation" (2 stages). A goal is a conjunction of sub-goals; episode progress is the mean of
# the per-sub-goal stage fractions (additive / parallel aggregation).
_PICKPLACE_PREDS = ("on", "in")
_ARTICULATION_PREDS = ("open", "close", "turnon", "turnoff")


def _is_push_subgoal(pred_args: list, is_push_task: bool) -> bool:
    """A pick-place predicate is "push-style" (no grasp) when the target is a table floor zone
    (e.g. ``main_table_stove_front_region``) or the task language starts with "push"."""
    target = str(pred_args[1]) if len(pred_args) >= 2 else ""
    return is_push_task or target.startswith("main_table_")


class _SubGoal:
    """One conjunct of the BDDL goal plus its ordered stage list and latched progress."""

    def __init__(self, pred: str, args: list, family: str, stages: list):
        self.pred = pred
        self.args = list(args)
        self.family = family
        self.stages = list(stages)
        self.max_reached = 0  # latched highest stage index reached this episode (0..len(stages))


class StageEvaluator:
    """Computes partial / progress scores for a LIBERO episode from ground-truth sim state.

    Purely client/sim-side: reads object & gripper state and LIBERO's own predicate / grasp helpers.
    The final stage of every sub-goal is the exact BDDL predicate, so reaching all final stages is
    identical to ``env._check_success()`` (progress == 1.0 <=> binary success).

    Latching is "strict ordering + soft credit": intermediate stages must be reached in order, but a
    satisfied predicate (the goal) credits all stages at once, and grasp implies "approached".

    Stage templates by sub-goal family:
      * pickplace (on/in onto an object/container) : approach_src -> grasp_src -> approach_tgt -> done
      * push (on/in into a table floor zone, or "push ..." task) : approach_src -> near_tgt -> done
      * articulation (open/close/turnon/turnoff)   : approach -> done
    """

    def __init__(self, env, approach_dist: float = 0.07, place_dist: float = 0.12, task_description: str = ""):
        self.inner = env.env  # underlying BDDLBaseDomain (ControlEnv wraps it as .env)
        self.d_app = approach_dist
        self.d_tgt = place_dist
        self.subgoals = []
        is_push_task = str(task_description).strip().lower().startswith("push")
        for conj in self.inner.parsed_problem["goal_state"]:
            pred = conj[0]
            pred_args = conj[1:]
            if pred in _PICKPLACE_PREDS and _is_push_subgoal(pred_args, is_push_task):
                # Push: the object is shoved into a floor zone, never grasped -> drop the grasp stage.
                stages = ["approach_src", "near_tgt", "done"]
                family = "push"
            elif pred in _PICKPLACE_PREDS:
                stages = ["approach_src", "grasp_src", "approach_tgt", "done"]
                family = "pickplace"
            else:  # open / close / turnon / turnoff (or anything unknown -> treat as 2-stage)
                stages = ["approach", "done"]
                family = "articulation"
            self.subgoals.append(_SubGoal(pred, pred_args, family, stages))

    def reset(self) -> None:
        for sg in self.subgoals:
            sg.max_reached = 0

    def _pos(self, name: str) -> np.ndarray:
        # Works for movable objects (body xpos), region sites (site xpos) and fixtures.
        return np.asarray(self.inner.object_states_dict[name].get_geom_state()["pos"], dtype=np.float64)

    def _grasped(self, name: str) -> bool:
        try:
            return bool(
                self.inner._check_grasp(
                    gripper=self.inner.robots[0].gripper,
                    object_geoms=self.inner.objects_dict[name],
                )
            )
        except Exception:  # noqa: BLE001 - object may not be a movable graspable object
            return False

    def _pred_true(self, sg: "_SubGoal") -> bool:
        try:
            return bool(self.inner._eval_predicate([sg.pred] + sg.args))
        except Exception:  # noqa: BLE001
            return False

    def _instant_stage(self, sg: "_SubGoal", obs: dict) -> int:
        """Highest stage index (1..K) currently consistent with progress; 0 if none."""
        if self._pred_true(sg):
            return len(sg.stages)  # goal satisfied -> soft-credit every stage
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        if sg.family == "pickplace":
            src, tgt = sg.args[0], sg.args[1]
            grasp = self._grasped(src)
            stage = 0
            try:
                if grasp or np.linalg.norm(eef - self._pos(src)) < self.d_app:
                    stage = 1  # approached source (grasp implies approached)
            except Exception:  # noqa: BLE001
                pass
            if grasp:
                stage = 2  # grasped source
            try:
                # Approaching the target only counts once we are (or have been) holding the source.
                if (grasp or sg.max_reached >= 2) and np.linalg.norm(
                    self._pos(src) - self._pos(tgt)
                ) < self.d_tgt:
                    stage = 3
            except Exception:  # noqa: BLE001
                pass
            return stage
        if sg.family == "push":
            src, tgt = sg.args[0], sg.args[1]
            stage = 0
            near_src = False
            try:
                near_src = np.linalg.norm(eef - self._pos(src)) < self.d_app
            except Exception:  # noqa: BLE001
                pass
            if near_src:
                stage = 1  # gripper reached the object to push
            try:
                # Object shoved toward the target zone (no grasp); strict order: after approach.
                if (near_src or sg.max_reached >= 1) and np.linalg.norm(
                    self._pos(src) - self._pos(tgt)
                ) < self.d_tgt:
                    stage = 2
            except Exception:  # noqa: BLE001
                pass
            return stage
        # articulation: single "approach the fixture/region" stage before the exact predicate.
        try:
            if np.linalg.norm(eef - self._pos(sg.args[0])) < self.d_tgt:
                return 1
        except Exception:  # noqa: BLE001
            pass
        return 0

    def update(self, obs: dict) -> None:
        for sg in self.subgoals:
            cur = self._instant_stage(sg, obs)
            if cur > sg.max_reached:
                sg.max_reached = cur

    def episode_progress(self) -> float:
        if not self.subgoals:
            return 1.0
        return float(np.mean([sg.max_reached / len(sg.stages) for sg in self.subgoals]))

    def episode_max_stages(self) -> list:
        return [sg.max_reached for sg in self.subgoals]

    def template(self) -> list:
        return [{"pred": sg.pred, "args": sg.args, "stages": sg.stages} for sg in self.subgoals]


def _summarize_progress(template: list, ep_progress: list, ep_max_stages: list) -> dict:
    """Aggregate per-episode stage records for one task into reach-fractions + mean progress."""
    n_eps = max(len(ep_progress), 1)
    subgoals = []
    for j, tmpl in enumerate(template):
        n_stages = len(tmpl["stages"])
        reach_fraction = [
            sum(1 for ms in ep_max_stages if ms[j] >= k) / n_eps for k in range(1, n_stages + 1)
        ]
        subgoals.append({**tmpl, "reach_fraction": reach_fraction})
    return {
        "mean_episode_progress": float(np.mean(ep_progress)) if ep_progress else 0.0,
        "subgoals": subgoals,
        "episode_progress": ep_progress,
        "episode_max_stages": ep_max_stages,
    }


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

        stage_eval = (
            StageEvaluator(env, args.approach_dist, args.place_dist, task_description=task_description)
            if args.progress_scores
            else None
        )
        task_ep_progress, task_ep_max_stages = [], []

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
            if stage_eval is not None:
                stage_eval.reset()
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
                    if stage_eval is not None:
                        stage_eval.update(obs)
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

            if stage_eval is not None:
                task_ep_progress.append(stage_eval.episode_progress())
                task_ep_max_stages.append(stage_eval.episode_max_stages())

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
        task_entry = {
            "task_id": task_id,
            "task_description": task_description,
            "episodes": task_episodes,
            "successes": task_successes,
            "success_rate": task_sr,
        }
        if stage_eval is not None:
            task_entry["progress"] = _summarize_progress(
                stage_eval.template(), task_ep_progress, task_ep_max_stages
            )
        per_task_metrics.append(task_entry)
        # Persist metrics incrementally so a crash mid-run still leaves partial results.
        _write_metrics(metrics_path, args, per_task_metrics, total_episodes, total_successes, t_start)
        if stage_eval is not None:
            logging.info(
                "Task %d success rate: %.3f | mean progress: %.3f",
                task_id, task_sr, task_entry["progress"]["mean_episode_progress"],
            )
        else:
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
    # Overall mean progress = episode-weighted mean across all tasks scored so far.
    all_ep_progress = [
        p for tm in per_task_metrics if "progress" in tm for p in tm["progress"]["episode_progress"]
    ]
    if all_ep_progress:
        payload["overall_mean_progress"] = float(np.mean(all_ep_progress))
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
