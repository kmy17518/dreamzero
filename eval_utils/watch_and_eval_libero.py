"""Watch a DreamZero LIBERO training run and auto-evaluate each new checkpoint in the LIBERO sim.

For every new ``checkpoint-N`` that appears under the training ``OUTPUT_DIR``, this:
  1. starts the DreamZero policy server (``eval_utils/serve_dreamzero_libero.py``, this
     ``dreamzero`` env) on a dedicated GPU, pointed at that checkpoint,
  2. runs the LIBERO sim client (``eval_utils/run_libero_eval.py``, the ``dreamzero_libero`` env),
  3. parses the resulting ``metrics.json`` (overall + per-task success rate), and
  4. logs the success rate to the SAME wandb run as training (resume by run id, custom
     ``eval/ckpt_step`` x-axis).

Retention (when --keep-best-n > 0): the watcher manages checkpoints IN PLACE in OUTPUT_DIR, keeping
``latest-L`` (most recent, by step) UNION ``best-M`` (highest eval success). Everything else is
deleted. Because the kept checkpoints are the originals, they include the DeepSpeed optimizer state
(resumable) and there is no duplication (a best ckpt that is also among the latest is kept once).
The watcher is the sole pruner, so launch training with a high ``save_total_limit`` (backstop only).

Durability (when --upload-repo is set): the best-M set is mirrored to a HuggingFace Hub repo in a
background thread -- uploaded when a checkpoint enters best-M, deleted from the repo when evicted.

Runs in the ``dreamzero`` env. The LIBERO client is launched via the ``dreamzero_libero`` env python.

Example:
  python eval_utils/watch_and_eval_libero.py \
      --output-dir ./checkpoints/dreamzero_libero_wan22 --server-gpu 7 \
      --num-trials-per-task 3 --keep-best-n 3 --keep-latest-n 5 \
      --upload-repo kmy17518/dreamzero-libero-best
"""

import argparse
import glob
import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[watch-eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ckpt_step(path: str):
    m = re.match(r".*checkpoint-(\d+)$", path)
    return int(m.group(1)) if m else None


def find_all_checkpoints(output_dir: str):
    """All checkpoint-N dirs (complete or not), sorted by step."""
    out = []
    for d in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        s = _ckpt_step(d)
        if s is not None and os.path.isdir(d):
            out.append((s, d))
    return sorted(out)


def find_complete_checkpoints(output_dir: str):
    """checkpoint-N dirs that are fully written + servable, sorted by step."""
    found = []
    for s, d in find_all_checkpoints(output_dir):
        needed = [
            os.path.join(d, "config.json"),
            os.path.join(d, "trainer_state.json"),  # written last by HF -> dir complete
            os.path.join(d, "experiment_cfg", "conf.yaml"),
        ]
        if not all(os.path.isfile(p) for p in needed):
            continue
        has_weights = (
            os.path.isfile(os.path.join(d, "model.safetensors"))
            or os.path.isfile(os.path.join(d, "model.safetensors.index.json"))
            or bool(glob.glob(os.path.join(d, "*.safetensors")))
            or bool(glob.glob(os.path.join(d, "pytorch_model*.bin")))
        )
        if has_weights:
            found.append((s, d))
    return found


def port_is_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def read_wandb_config(output_dir: str):
    p = os.path.join(output_dir, "wandb_config.json")
    if os.path.isfile(p):
        try:
            d = json.load(open(p))
            return d.get("project") or None, d.get("run_id") or None
        except Exception:
            pass
    return None, None


# ------------------------------ persistent state ------------------------------
_state_lock = threading.Lock()


def load_state(state_path: str):
    if os.path.isfile(state_path):
        try:
            d = json.load(open(state_path))
            results = {int(k): float(v) for k, v in d.get("results", {}).items()}
            return (
                set(d.get("evaluated", [])),
                set(d.get("failed", [])),
                results,
                set(d.get("uploaded", [])),
            )
        except Exception:
            pass
    return set(), set(), {}, set()


def save_state(state_path, evaluated, failed, results, uploaded):
    with _state_lock:
        os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(
                {
                    "evaluated": sorted(evaluated),
                    "failed": sorted(failed),
                    "results": {str(k): results[k] for k in sorted(results)},
                    "uploaded": sorted(uploaded),
                },
                f,
                indent=2,
            )


# ------------------------------ retention ------------------------------
def compute_retention(output_dir, keep_latest, keep_best, results):
    """Return (keep_steps:set, best_steps:list) for latest-L UNION best-M over existing ckpts."""
    all_ck = find_all_checkpoints(output_dir)
    if not all_ck:
        return set(), []
    existing = {s for s, _ in all_ck}
    newest = max(existing)
    latest_keep = set(sorted(existing, reverse=True)[:keep_latest])
    ranked = sorted(
        ((s, results[s]) for s in results if s in existing),
        key=lambda kv: (-kv[1], -kv[0]),
    )
    best_steps = [s for s, _ in ranked[:keep_best]]
    keep = latest_keep | set(best_steps) | {newest}
    return keep, best_steps


def apply_retention(output_dir, keep_latest, keep_best, results, manifest_path):
    """Delete complete checkpoints not in latest-L UNION best-M (in place). Never deletes the
    newest or an in-progress (incomplete) checkpoint. Returns best_steps."""
    keep, best_steps = compute_retention(output_dir, keep_latest, keep_best, results)
    if not keep:
        return best_steps
    complete = {s for s, _ in find_complete_checkpoints(output_dir)}
    for s, path in find_all_checkpoints(output_dir):
        if s in keep or s not in complete:
            continue
        log(f"retention: deleting checkpoint-{s} (not in latest-{keep_latest} or best-{keep_best})")
        shutil.rmtree(path, ignore_errors=True)
    try:
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "keep_latest_n": keep_latest,
                    "keep_best_n": keep_best,
                    "best": [{"step": s, "success_rate": results.get(s)} for s in best_steps],
                    "kept_steps": sorted(keep),
                },
                f,
                indent=2,
            )
    except Exception:
        pass
    return best_steps


# ------------------------------ HF Hub mirror (background) ------------------------------
class HfUploader(threading.Thread):
    """Mirror best checkpoints to a HuggingFace Hub repo in the background (upload on add, delete
    on evict). Updates the `uploaded` set + state via the provided callbacks."""

    def __init__(self, repo_id, repo_type, model_only, private, token,
                 mark_uploaded, mark_deleted):
        super().__init__(daemon=True)
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.model_only = model_only
        self.mark_uploaded = mark_uploaded
        self.mark_deleted = mark_deleted
        self.q: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
        log(f"upload: mirroring best ckpts to {repo_type} repo '{repo_id}' "
            f"({'model-only' if model_only else 'full incl. optimizer'})")

    def enqueue(self, action, step, src=None):
        self.q.put((action, step, src))

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                action, step, src = self.q.get(timeout=2)
            except queue.Empty:
                continue
            try:
                if action == "up":
                    ignore = (["global_step*", "rng_state_*", "optimizer*", "scheduler.pt"]
                              if self.model_only else None)
                    log(f"upload: pushing checkpoint-{step} -> {self.repo_id} (queued: {self.q.qsize()})")
                    self.api.upload_folder(
                        folder_path=src,
                        path_in_repo=f"checkpoint-{step}",
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        ignore_patterns=ignore,
                        commit_message=f"best checkpoint-{step}",
                    )
                    self.mark_uploaded(step)
                    log(f"upload: done checkpoint-{step}")
                elif action == "del":
                    try:
                        self.api.delete_folder(
                            path_in_repo=f"checkpoint-{step}",
                            repo_id=self.repo_id,
                            repo_type=self.repo_type,
                            commit_message=f"evict checkpoint-{step}",
                        )
                    except Exception as e:  # noqa: BLE001
                        log(f"upload: delete checkpoint-{step} (likely already gone): {e}")
                    self.mark_deleted(step)
                    log(f"upload: removed checkpoint-{step} from {self.repo_id}")
            except Exception as e:  # noqa: BLE001
                log(f"upload: FAILED {action} checkpoint-{step}: {e}")
            finally:
                self.q.task_done()


# ------------------------------ server / client ------------------------------
def kill_process_group(proc: subprocess.Popen):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def start_server(args, ckpt_dir: str, server_log_path: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.server_gpu)
    env.setdefault("MASTER_PORT", str(29500 + (args.port % 1000)))
    cmd = [
        args.dreamzero_python,
        "eval_utils/serve_dreamzero_libero.py",
        "--model_path", ckpt_dir,
        "--embodiment_tag", "libero_sim",
        "--tokenizer_path", args.tokenizer_path,
        "--host", "127.0.0.1",
        "--port", str(args.port),
    ]
    log(f"starting server: CUDA_VISIBLE_DEVICES={args.server_gpu} {' '.join(cmd)}")
    fh = open(server_log_path, "w")
    return subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_server_ready(proc, port, log_path, timeout) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            log(f"server exited early (code {proc.returncode}) -- see {log_path}")
            return False
        if port_is_open("127.0.0.1", port):
            log(f"server ready on port {port} after {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"server did not become ready within {timeout}s")
    return False


def run_client(args, step, eval_out_root) -> str | None:
    videos_dir = os.path.join(eval_out_root, f"checkpoint-{step}", "videos")
    metrics_path = os.path.join(eval_out_root, f"checkpoint-{step}", "metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    env = os.environ.copy()
    env["MUJOCO_GL"] = args.mujoco_gl
    if args.mujoco_gl == "egl":
        # Pin EGL rendering to the dedicated eval GPU. The default EGL device 0 may be saturated by
        # training on a busy multi-GPU node, which makes MuJoCo abort (SIGABRT / exit -6).
        env["MUJOCO_EGL_DEVICE_ID"] = str(args.server_gpu)
    cmd = [
        args.libero_python, "eval_utils/run_libero_eval.py",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--task-suite-name", args.task_suite_name,
        "--num-trials-per-task", str(args.num_trials_per_task),
        "--video-out-path", videos_dir,
        "--metrics-out-path", metrics_path,
    ]
    if args.max_tasks > 0:
        cmd += ["--max-tasks", str(args.max_tasks)]
    if args.max_steps_override > 0:
        cmd += ["--max-steps-override", str(args.max_steps_override)]
    if not args.save_videos:
        cmd += ["--no-save-videos"]
    log(f"running client (dreamzero_libero): {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if ret.returncode != 0:
        log(f"client exited with code {ret.returncode}")
    return metrics_path if os.path.isfile(metrics_path) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, help="training OUTPUT_DIR to watch")
    ap.add_argument("--server-gpu", default="7")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tokenizer-path", default="./checkpoints/umt5-xxl")
    ap.add_argument("--task-suite-name", default="libero_spatial")
    ap.add_argument("--num-trials-per-task", type=int, default=10)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--max-steps-override", type=int, default=0)
    ap.add_argument("--save-videos", action="store_true", default=True)
    ap.add_argument("--no-save-videos", dest="save_videos", action="store_false")
    ap.add_argument("--poll-interval", type=int, default=60)
    ap.add_argument("--server-ready-timeout", type=int, default=900)
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--all", dest="latest_only", action="store_false", default=True,
                    help="evaluate every checkpoint (default: only the newest un-evaluated one)")
    ap.add_argument("--exit-when-done", action="store_true")
    ap.add_argument("--mujoco-gl", default="egl")
    ap.add_argument("--eval-out-root", default=None)
    ap.add_argument("--dreamzero-python", default=sys.executable)
    ap.add_argument("--libero-python", default="/root/miniconda3/envs/dreamzero_libero/bin/python")
    # retention (in-place: keep latest-L UNION best-M, full incl. optimizer, no copies)
    ap.add_argument("--keep-best-n", type=int, default=0,
                    help="keep the best-N (by eval success) checkpoints in place (>0 enables "
                         "watcher-managed retention; the watcher becomes the sole pruner)")
    ap.add_argument("--keep-latest-n", type=int, default=5,
                    help="also always keep the latest-N checkpoints by step (default 5)")
    # wandb
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-run-id", default=None)
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--separate-run", action="store_true")
    # HF Hub upload (durable mirror of best-M)
    ap.add_argument("--upload-repo", default=None,
                    help="HF Hub repo id (e.g. user/dreamzero-libero-best) to mirror best-M into")
    ap.add_argument("--upload-repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--upload-model-only", action="store_true",
                    help="upload model-only (no optimizer) to the Hub -- much smaller/faster "
                         "(default uploads the full, resumable checkpoint)")
    ap.add_argument("--upload-public", action="store_true", help="make the Hub repo public (default private)")
    args = ap.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    eval_out_root = args.eval_out_root or os.path.join(output_dir, "eval_outputs")
    state_path = os.path.join(eval_out_root, "_watch_eval_state.json")
    manifest_path = os.path.join(output_dir, "best_checkpoints.json")
    evaluated, failed, results, uploaded = load_state(state_path)

    cfg_project, cfg_run_id = read_wandb_config(output_dir)
    project = args.wandb_project or cfg_project or os.environ.get("WANDB_PROJECT") or "dreamzero_libero"
    run_id = args.wandb_run_id or cfg_run_id
    if args.wandb_mode != "disabled" and not run_id:
        log("WARNING: no wandb run_id found (launch training with a fixed WANDB_RUN_ID to log eval "
            "into the same run). Falling back to output-dir name.")
        run_id = os.path.basename(output_dir)
    if args.separate_run and run_id:
        run_id = f"{run_id}-eval"

    wandb_run = None
    if args.wandb_mode != "disabled":
        os.environ["WANDB_MODE"] = args.wandb_mode
        import wandb
        wandb_run = wandb.init(
            project=project, id=run_id, resume="allow",
            name=(run_id if args.separate_run else None),
            job_type="libero_eval",
        )
        wandb.define_metric("eval/ckpt_step")
        wandb.define_metric("eval/*", step_metric="eval/ckpt_step")

    # background HF uploader
    uploader = None
    if args.upload_repo:
        def _mark_uploaded(step):
            uploaded.add(step)
            save_state(state_path, evaluated, failed, results, uploaded)

        def _mark_deleted(step):
            uploaded.discard(step)
            save_state(state_path, evaluated, failed, results, uploaded)

        uploader = HfUploader(
            repo_id=args.upload_repo, repo_type=args.upload_repo_type,
            model_only=args.upload_model_only, private=not args.upload_public,
            token=os.environ.get("HF_TOKEN"),
            mark_uploaded=_mark_uploaded, mark_deleted=_mark_deleted,
        )
        uploader.start()

    log(f"watching {output_dir} | server-gpu={args.server_gpu} | suite={args.task_suite_name} "
        f"trials/task={args.num_trials_per_task} max_tasks={args.max_tasks or 'all'} | "
        f"latest_only={args.latest_only} | keep_latest={args.keep_latest_n} keep_best={args.keep_best_n}"
        + (f" | upload->{args.upload_repo}" if uploader else ""))

    def sync_uploads(best_steps):
        if uploader is None:
            return
        # HF mirror = best-N (by success rate) UNION the latest complete checkpoint, so the most
        # recent training state is always resumable from the Hub even before it is (or if it never
        # becomes) a top scorer. Anything previously uploaded that is no longer in this set is removed.
        upload_set = set(best_steps)
        complete = [s for s, _ in find_complete_checkpoints(output_dir)]
        if complete:
            upload_set.add(max(complete))
        for s in sorted(upload_set):
            if s in uploaded:
                continue
            src = os.path.join(output_dir, f"checkpoint-{s}")
            if os.path.isdir(src):
                uploader.enqueue("up", s, src)
        for s in sorted(uploaded - upload_set):
            uploader.enqueue("del", s)

    try:
        while True:
            # 1) prune in place to latest-L UNION best-M (every cycle; cheap)
            if args.keep_best_n > 0:
                best_steps = apply_retention(
                    output_dir, args.keep_latest_n, args.keep_best_n, results, manifest_path)
                sync_uploads(best_steps)

            # 2) evaluate new checkpoint(s)
            cks = [(s, p) for (s, p) in find_complete_checkpoints(output_dir)
                   if s >= args.min_step and s not in evaluated and s not in failed]
            if cks:
                todo = [cks[-1]] if args.latest_only else cks
                if args.latest_only and len(cks) > 1:
                    for s, _ in cks[:-1]:
                        evaluated.add(s)  # mark skipped so we don't reconsider
                    log(f"latest-only: skipping {[s for s,_ in cks[:-1]]}, evaluating {todo[0][0]}")
                    save_state(state_path, evaluated, failed, results, uploaded)
                for step, ckpt_dir in todo:
                    if not os.path.isdir(ckpt_dir):
                        failed.add(step); save_state(state_path, evaluated, failed, results, uploaded); continue
                    log(f"=== evaluating checkpoint-{step} ===")
                    server_log = os.path.join(eval_out_root, f"checkpoint-{step}", "server.log")
                    os.makedirs(os.path.dirname(server_log), exist_ok=True)
                    proc = start_server(args, ckpt_dir, server_log)
                    if not wait_server_ready(proc, args.port, server_log, args.server_ready_timeout):
                        kill_process_group(proc)
                        failed.add(step); save_state(state_path, evaluated, failed, results, uploaded); continue
                    metrics_path = None
                    try:
                        metrics_path = run_client(args, step, eval_out_root)
                    finally:
                        kill_process_group(proc)
                    if not metrics_path:
                        log(f"no metrics for checkpoint-{step}")
                        failed.add(step); save_state(state_path, evaluated, failed, results, uploaded); continue
                    metrics = json.load(open(metrics_path))
                    sr = float(metrics.get("overall_success_rate", 0.0))
                    results[step] = sr
                    evaluated.add(step)
                    log(f"checkpoint-{step}: overall success {sr:.3f} "
                        f"({metrics.get('total_successes')}/{metrics.get('total_episodes')})")
                    if wandb_run is not None:
                        import wandb
                        payload = {
                            "eval/ckpt_step": step,
                            "eval/success_rate": sr,
                            "eval/total_episodes": metrics.get("total_episodes", 0),
                            "eval/total_successes": metrics.get("total_successes", 0),
                        }
                        for t in metrics.get("per_task", []):
                            payload[f"eval/task{t['task_id']}_success_rate"] = t.get("success_rate", 0.0)
                        wandb.log(payload)
                    save_state(state_path, evaluated, failed, results, uploaded)
                    # re-run retention now that we have a new result (may promote/evict best)
                    if args.keep_best_n > 0:
                        best_steps = apply_retention(
                            output_dir, args.keep_latest_n, args.keep_best_n, results, manifest_path)
                        sync_uploads(best_steps)
            else:
                finished = os.path.isfile(os.path.join(output_dir, "config.json"))
                if finished and args.exit_when_done:
                    if uploader is not None:
                        log("waiting for pending uploads to finish...")
                        uploader.q.join()
                    log("training finished and nothing left to evaluate; exiting.")
                    break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        log("interrupted; exiting.")
    finally:
        if uploader is not None:
            uploader.stop()
        # Do NOT wandb.finish() when resuming the training run (it would finalize the shared run).
        if wandb_run is not None and args.separate_run:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
