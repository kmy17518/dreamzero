# LIBERO — Final Reproduction Guide (run AS-IS from an empty instance)

Compact, self-contained recipe to stand up **exactly the training + eval running now**:
resume DreamZero‑LIBERO (Wan2.2‑TI2V‑5B backbone, full fine‑tune, **per‑device batch size 1**)
from the HF mirror, with **7‑GPU training (GPUs 0–6) + 1‑GPU auto eval/upload watcher (GPU 7)**
on an 8‑GPU box.

- Verified on **8× B300 (Blackwell, `sm_103a`)**. **Hopper (H100/H200, `sm_90`) differences are called out inline** (mainly the eval `torch.compile` toggle).
- Assumes repo at `/root/dreamzero`, conda at `/root/miniconda3`, branch **`libero`**.
- Full reference (background, design decisions): `docs/LIBERO.md`. This file is the practical "do this".

```bash
cd /root/dreamzero && git checkout libero      # all LIBERO code/scripts live on this branch
```

---

## 0. Prereqs

- 8 GPUs (H100/H200/B300). Training uses **0–6**, eval server uses **7**.
- CUDA 12.x toolkit at `/usr/local/cuda` (provides `nvcc`). `tmux`, `git`.
- `/root/dreamzero/.env` (git‑ignored) containing:
  ```
  HF_TOKEN=hf_...          # WRITE scope; access to kmy17518/dreamzero-libero-best
  WANDB_API_KEY=...
  ```
  > **HF storage/billing:** full checkpoints are ~130 GB each. Uploading more than the included
  > quota requires **pay‑as‑you‑go billing enabled** (Settings → Billing → automatic credit recharge),
  > else LFS uploads 403 with *"setup automatic credit recharge"*. The watcher caps usage (see §5).
- OSMesa for headless MuJoCo rendering during eval:
  ```bash
  apt-get update -y && apt-get install -y libosmesa6 libgl1-mesa-glx
  ```
- Know your GPU arch (decides the eval compile toggle):
  ```bash
  python -c "import torch; print(torch.cuda.get_device_capability())"
  # (10, x) -> Blackwell (B300)  -> EVAL EAGER (set TORCHDYNAMO_DISABLE=1, see §3d/§4)
  # (9, 0)  -> Hopper (H100/H200) -> EVAL COMPILED (omit it; faster)
  ```

---

## 1. Conda environments

**Miniconda (if none):**
```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /root/miniconda.sh
bash /root/miniconda.sh -b -p /root/miniconda3
/root/miniconda3/bin/conda init bash && source ~/.bashrc
```

**Env 1 — `dreamzero` (training + policy serving, GPU, py3.11):**
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n dreamzero python=3.11 -y -c conda-forge --override-channels
conda activate dreamzero
python -m ensurepip --upgrade                  # conda-forge python ships without pip
cd /root/dreamzero
export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129   # torch 2.8 cu129
MAX_JOBS=96 python -m pip install --no-build-isolation flash-attn                     # usually a prebuilt 2.8.3 wheel
python -m pip install hf_transfer             # fast HF downloads
```
> - **flash-attn on B300:** the prebuilt 2.8.3 wheel's `sm_100` kernels run fine on `sm_103`. If it
>   instead *compiles*, force the arch: `TORCH_CUDA_ARCH_LIST="10.0+PTX"` (Hopper: `"9.0+PTX"`).
>   `nvcc` 12.8 cannot target `sm_103`/`compute_103`; `sm_100`+PTX covers the B300.
> - **Do NOT** `pip install -U huggingface_hub`. `pip install -e .` already pins `huggingface_hub==0.36.2`
>   (which has the `hf` CLI **and** `permanently_delete_lfs_files`); upgrading it breaks `transformers`/`tokenizers`.

**Env 2 — `dreamzero_libero` (LIBERO MuJoCo sim client, CPU, py3.10):**
```bash
conda create -n dreamzero_libero python=3.10 pip -y -c conda-forge --override-channels
conda activate dreamzero_libero
python -m pip install "setuptools==65.5.0" "wheel==0.38.4" "pip==23.3.2"
python -m pip install "numpy==1.24.4"
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "robosuite==1.4.1" "mujoco==3.2.3" "bddl==1.0.1" "easydict==1.9" \
            "opencv-python==4.6.0.66" Pillow "matplotlib==3.5.3"
python -m pip install "gym==0.25.2" --no-build-isolation
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /root/LIBERO
python -m pip install -e /root/LIBERO --no-deps
python -m pip install websockets msgpack msgpack-numpy openpi-client tqdm tyro imageio \
            imageio-ffmpeg "hydra-core==1.2.0" termcolor future cloudpickle
mkdir -p ~/.libero && cat > ~/.libero/config.yaml <<'YAML'
benchmark_root: /root/LIBERO/libero/libero
bddl_files:     /root/LIBERO/libero/libero/bddl_files
init_states:    /root/LIBERO/libero/libero/init_files
datasets:       /root/LIBERO/libero/datasets
assets:         /root/LIBERO/libero/libero/assets
YAML
```

---

## 2. Download backbone checkpoints + dataset

```bash
conda activate dreamzero && cd /root/dreamzero && set -a; . ./.env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p checkpoints data

# (a) Wan2.2-TI2V-5B backbone (~34 GB: DiT, Wan2.2_VAE.pth, T5 encoder, umt5 tokenizer)
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
# (b) CLIP image encoder from Wan2.1 (~4.5 GB; Wan2.2-TI2V-5B doesn't ship it)
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
# (c) umt5-xxl tokenizer (bundled inside Wan2.2 — just copy it out)
mkdir -p ./checkpoints/umt5-xxl && cp ./checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl/* ./checkpoints/umt5-xxl/
# (d) LIBERO dataset -> convert to DreamZero MP4 LeRobot format (1693 eps / 273k frames, ~11 GB)
hf download physical-intelligence/libero --repo-type dataset --local-dir ./data/libero_raw_lerobot
python scripts/data/convert_libero_to_dreamzero.py \
    --src data/libero_raw_lerobot --dst data/libero_lerobot --num-workers 64
```

---

## 3. Resume training (7 GPUs) + eval watcher (1 GPU) in tmux

### 3a. Download the checkpoint to resume from
Pull the **highest‑numbered** checkpoint from the mirror **into the OUTPUT_DIR** (these are FULL
checkpoints incl. the DeepSpeed `global_step*` optimizer state, so training resumes exactly):
```bash
conda activate dreamzero && cd /root/dreamzero && set -a; . ./.env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
export OUTPUT_DIR=$PWD/checkpoints/dreamzero_libero_resume
mkdir -p "$OUTPUT_DIR"
LATEST=$(python -c "
from huggingface_hub import HfApi; import os
a=HfApi(token=os.environ['HF_TOKEN'])
d={s.rfilename.split('/')[0] for s in a.repo_info('kmy17518/dreamzero-libero-best',repo_type='model').siblings if s.rfilename.startswith('checkpoint-')}
print(max(int(x.split('-')[1]) for x in d))")
echo "Resuming from checkpoint-$LATEST"
hf download kmy17518/dreamzero-libero-best --include "checkpoint-$LATEST/*" --local-dir "$OUTPUT_DIR"
```
> **CRITICAL — GPU count must match the saved ZeRO ranks.** The checkpoints were saved with
> DeepSpeed ZeRO‑2 across **7 data‑parallel ranks** (`global_step*/bf16_zero_pp_rank_0..6_*`), so you
> **must resume with exactly `NUM_GPUS=7`** (GPUs 0–6). Using 8 would fail the optimizer‑state load.
> Also: `OUTPUT_DIR` must contain only `checkpoint-N/` and **no top‑level `config.json`** (a top‑level
> `config.json` makes the launcher treat the run as *finished* and skip). Resume is automatic (no flag).

### 3b. Create the tmux session (two panes)
```bash
tmux kill-session -t libero 2>/dev/null; tmux new-session -d -s libero -x 250 -y 50
tmux split-window -h -t libero:0
tmux select-layout -t libero:0 even-horizontal
# pane 0 (left) = training, pane 1 (right) = eval watcher
```

### 3c. Pane 0 — TRAINING (GPUs 0–6, bs=1). Paste as ONE line (tmux can drop inline VAR=val on multi-line):
```bash
tmux send-keys -t libero:0.0 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate dreamzero && cd /root/dreamzero && set -a; . ./.env; set +a && export HYDRA_FULL_ERROR=1 WANDB_RUN_ID=dreamzero_libero_run_v1 && NUM_GPUS=7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 SAVE_TOTAL_LIMIT=100000 OUTPUT_DIR=$PWD/checkpoints/dreamzero_libero_resume LIBERO_DATA_ROOT=$PWD/data/libero_lerobot PER_DEVICE_BATCH_SIZE=1 MAX_STEPS=100000 SAVE_STEPS=1000 SAVE_STRATEGY=steps TRAIN_ARCH=full PYTHON_BIN=$(which python) bash scripts/train/libero_training_wan22.sh 2>&1 | tee /root/train.log' C-m
```
> - `WANDB_RUN_ID` must be **fresh & never‑deleted** (wandb 409s on reused/deleted ids). The eval
>   watcher logs to the sibling run `<WANDB_RUN_ID>-eval`, so use the **same id** in both panes.
> - `PER_DEVICE_BATCH_SIZE=1` only — the `libero` branch supports bs=1 (the model's action‑head loss
>   assumes one sample/device). Grow global batch via more GPUs / `gradient_accumulation_steps`.

### 3d. Pane 1 — EVAL WATCHER (GPU 7). **Blackwell vs Hopper differs here:**

**Blackwell (B300, `sm_103a`) — run the policy server EAGER:**
```bash
tmux send-keys -t libero:0.1 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate dreamzero && cd /root/dreamzero && export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 && OUTPUT_DIR=$PWD/checkpoints/dreamzero_libero_resume SERVER_GPU=7 TRIALS=3 MAX_TASKS=3 KEEP_BEST=2 KEEP_LATEST=3 MILESTONE_INTERVAL=5000 UPLOAD_REPO=kmy17518/dreamzero-libero-best bash scripts/eval/watch_eval_libero.sh --wandb-run-id dreamzero_libero_run_v1 2>&1 | tee /root/eval.log' C-m
```

**Hopper (H100/H200, `sm_90`) — omit the compile-disable (compile works and is faster):**
```bash
# same as above but DROP `export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 &&`
tmux send-keys -t libero:0.1 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate dreamzero && cd /root/dreamzero && OUTPUT_DIR=$PWD/checkpoints/dreamzero_libero_resume SERVER_GPU=7 TRIALS=3 MAX_TASKS=3 KEEP_BEST=2 KEEP_LATEST=3 MILESTONE_INTERVAL=5000 UPLOAD_REPO=kmy17518/dreamzero-libero-best bash scripts/eval/watch_eval_libero.sh --wandb-run-id dreamzero_libero_run_v1 2>&1 | tee /root/eval.log' C-m
```

> **Why the difference (important):** the policy server `torch.compile`s the text/image/VAE encoders.
> Inductor → Triton → `ptxas`. The CUDA 12.8 `ptxas` bundled with torch/triton **cannot target
> `sm_103a`** (B300) → inference aborts: *"ptxas fatal: Value 'sm_103a' is not defined"* → every
> episode errors (spurious 0% success). `TORCHDYNAMO_DISABLE=1` runs the server **eager** (identical
> results, no Triton). On **Hopper (`sm_90`)** `ptxas` supports the arch, so compile works and is
> faster — don't disable it. **Training is unaffected on both** (it already runs eager).
>
> Watcher behavior (both archs): serves each new `checkpoint-N` on GPU 7, runs the LIBERO sim eval,
> logs `eval/success_rate` (+ per‑task) to the sibling wandb run, and mirrors checkpoints to HF.
> Rendering defaults to `MUJOCO_GL=osmesa` (CPU) for robustness on a busy box; on a fully idle GPU
> set `MUJOCO_GL_BACKEND=egl` for faster GPU rendering. Knobs: `SERVER_GPU`, `TRIALS` (per task),
> `MAX_TASKS` (0 = all 10), `KEEP_BEST`, `KEEP_LATEST`, `MILESTONE_INTERVAL` (0 disables), `UPLOAD_REPO`,
> `UPLOAD_MODEL_ONLY=1` (~25 GB model‑only instead of ~130 GB full).

Attach to watch both: `tmux attach -t libero`  (logs: `/root/train.log`, `/root/eval.log`).

### 3e. Verify
```bash
grep -m1 "Resuming training from" /root/train.log    # training picked up the checkpoint
grep -m1 "success="              /root/eval.log       # an episode ran (NOT an InductorError/ptxas crash)
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader   # 0-6 busy, 7 = eval
```

---

## 4. Standalone eval (no watcher) — all task suites, all tasks, default trials

Two terminals. **Terminal A = policy server (GPU, `dreamzero`)**, **Terminal B = sim client (CPU, `dreamzero_libero`)**.
Point `--model_path` at any `checkpoint-N/` (or the top‑level finished model).

**Terminal A — server** (set `CKPT` to the checkpoint to score):
```bash
conda activate dreamzero && cd /root/dreamzero && set -a; . ./.env; set +a
export CKPT=$PWD/checkpoints/dreamzero_libero_resume/checkpoint-22000   # <-- edit
# Blackwell: prefix TORCHDYNAMO_DISABLE=1 (eager).  Hopper: drop it (compiled, faster).
TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES=7 python eval_utils/serve_dreamzero_libero.py \
    --model_path "$CKPT" --embodiment_tag libero_sim \
    --tokenizer_path ./checkpoints/umt5-xxl --port 8000
```

**Terminal B — client over all 5 suites, all tasks, default 50 trials/task:**
```bash
conda activate dreamzero_libero && cd /root/dreamzero
for SUITE in libero_spatial libero_object libero_goal libero_10 libero_90; do
  echo "==== $SUITE ===="
  MUJOCO_GL=osmesa python eval_utils/run_libero_eval.py \
      --host 0.0.0.0 --port 8000 --task-suite-name "$SUITE" \
      --num-trials-per-task 50 \
      --video-out-path ./eval_outputs/full/$SUITE/videos \
      --metrics-out-path ./eval_outputs/full/$SUITE/metrics.json
done
```
> - **No `--max-tasks`** ⇒ every task in the suite; **`--num-trials-per-task 50`** is the default.
> - Suites & episode caps: `libero_spatial`(220) `libero_object`(280) `libero_goal`(300) `libero_10`(520) `libero_90`(400).
> - This is **long** (5 suites × all tasks × 50 trials, ~2–2.5 min/episode). For a quick smoke test add
>   `--max-tasks 1 --num-trials-per-task 2 --max-steps-override 60`.
> - `metrics.json` (per‑task + overall success) is written incrementally; rollout MP4s land under `--video-out-path`.
> - On Blackwell keep `MUJOCO_GL=osmesa` if GPU 7 is shared; use `MUJOCO_GL=egl` only with a dedicated idle GPU.

---

## 5. Notes / gotchas (this iteration)

1. **Resume needs `NUM_GPUS` = saved ZeRO ranks (7).** Mismatch breaks optimizer‑state load.
2. **bs=1 only** on the `libero` branch.
3. **Eval compile: Blackwell eager (`TORCHDYNAMO_DISABLE=1`), Hopper compiled.** See §3d. Training eager on both.
4. **HF storage is billed across full commit history**, and `super_squash_history` does **not** free LFS
   blobs. The watcher reclaims for real via `permanently_delete_lfs_files()` on eviction. It keeps
   **best‑2 ∪ latest ∪ every `MILESTONE_INTERVAL` (5000) milestone** on the Hub permanently; milestones
   are kept locally only until uploaded (Hub = archive) so disk stays lean. Steady‑state HF usage ≈
   (#milestones + best‑2 + latest) × ~130 GB. Enable HF pay‑as‑you‑go billing or uploads 403.
5. **`WANDB_RUN_ID` must be fresh** (never previously created+deleted); training and watcher share it,
   eval logs to `<id>-eval`. In the wandb UI set the chart X‑axis to `train/global_step`.
6. **`.env` is git‑ignored** — never commit it.
7. **B300 perf:** bs=1 is latency‑bound, so a B300 step (~2.0 s) can be *slower* than H200 (~1.24 s) —
   the GPU draws only ~⅓ of its power at 99% "util". Raise throughput with a larger per‑device batch
   (needs `libero_distributed` branch), not by upgrading the GPU at bs=1.
