#!/bin/bash
# Launch the LIBERO checkpoint-eval watcher (runs in the `dreamzero` env).
#
# For every new checkpoint-N under OUTPUT_DIR it serves the policy on SERVER_GPU, runs the LIBERO
# sim eval (dreamzero_libero env), and logs the success rate to the SAME wandb run as training
# (resumed by the run id recorded in OUTPUT_DIR/wandb_config.json -- so launch training with a
# fixed WANDB_RUN_ID).
#
# Usage:
#   OUTPUT_DIR=$PWD/checkpoints/dreamzero_libero_wan22 SERVER_GPU=7 bash scripts/eval/watch_eval_libero.sh
# Extra flags are passed through, e.g.:
#   ... bash scripts/eval/watch_eval_libero.sh --num-trials-per-task 20 --max-tasks 5
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
set -a; . ./.env; set +a          # WANDB_API_KEY (+ HF_TOKEN)

OUTPUT_DIR=${OUTPUT_DIR:-"$PWD/checkpoints/dreamzero_libero_wan22"}

EXTRA=()
[ -n "$UPLOAD_REPO" ] && EXTRA+=(--upload-repo "$UPLOAD_REPO")
[ "${UPLOAD_MODEL_ONLY:-0}" = "1" ] && EXTRA+=(--upload-model-only)
# Eval logs to a sibling "<run>-eval" wandb run by default: a watcher and training cannot reliably
# share one *live* run (the second writer's points are silently dropped). Set SEPARATE_RUN=0 to
# attempt same-run logging (not recommended).
[ "${SEPARATE_RUN:-1}" = "1" ] && EXTRA+=(--separate-run)

exec python eval_utils/watch_and_eval_libero.py \
    --output-dir "$OUTPUT_DIR" \
    --server-gpu "${SERVER_GPU:-7}" \
    --task-suite-name "${TASK_SUITE:-libero_spatial}" \
    --num-trials-per-task "${TRIALS:-10}" \
    --max-tasks "${MAX_TASKS:-3}" \
    --keep-best-n "${KEEP_BEST:-0}" \
    --keep-latest-n "${KEEP_LATEST:-5}" \
    --upload-best-n "${UPLOAD_BEST:-2}" \
    --milestone-interval "${MILESTONE_INTERVAL:-0}" \
    --mujoco-gl "${MUJOCO_GL_BACKEND:-osmesa}" \
    "${EXTRA[@]}" \
    "$@"
