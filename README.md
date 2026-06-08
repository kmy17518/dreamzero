# CS224R Project: Do generated futures help robot policies through representation alignment or explicit conditioning?


**Motivation.** Many recent robot-learning systems predict a generated future — a future image, video, or
language sub-goal — and report performance gains, yet it is unclear *why* these futures help. This repo
isolates two mechanisms that are usually entangled: **representation alignment** (predicting the future
during training shapes a better internal representation) and **explicit conditioning** (the generated
future is fed into the action path as additional input).

These experiments are **WAM (World-Action Model) variants** — built on the **Wan2.2-TI2V-5B** video
backbone, with **DreamZero** used as reference — full-fine-tuned and evaluated on the
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark in the LIBERO MuJoCo simulator. A
WAM jointly predicts actions and video; each experiment below toggles how (and whether) the generated
future participates in training and eval.


---

## The four train/eval options

Each option is just a Hydra **action-head config**: the option name *is* the value you pass as
`model/dreamzero/action_head=<option>`, and the matching launcher sets it for you. All options share the
same dataset, backbone weights, two-environment eval harness, and checkpoint/eval watcher — they differ
only in this action-head config.

| Option | Action-head config | Train launcher |
|---|---|---|
| `action_loss_only` **(baseline)** | `action_loss_only` | `scripts/train/libero_training_wan22_action_loss_only_skip_noisy_video.sh` |
| `joint_action_loss_dynamics_loss` | `joint_action_loss_dynamics_loss` | `scripts/train/libero_training_wan22.sh` |
| `decoupled_action_loss_dynamics_loss` | `decoupled_action_loss_dynamics_loss` | `scripts/train/libero_training_wan22_decoupled.sh` |
| `use_generated_video_feedback` | `use_generated_video_feedback` | `scripts/train/libero_training_wan22_future_shift.sh` |

Mapped onto the two mechanisms: `action_loss_only` is the **baseline** — a pure action policy with no
future prediction; `decoupled_action_loss_dynamics_loss` adds the future/dynamics-prediction loss (with
the action→generated-video attention cut) to probe **representation alignment**; `joint_action_loss_dynamics_loss`
adds **explicit conditioning** on the latent video tokens (the action attends to the being-generated
latent video); and `use_generated_video_feedback` adds **explicit conditioning** on the generated video
frames (the model's own generated future is fed back into the action path).

See [`docs/`](docs/) for more information on:
[`GENERATED_VIDEO_FEEDBACK.md`](docs/GENERATED_VIDEO_FEEDBACK.md) (explicit conditioning) and
[`LIBERO_SOFT_EVAL.md`](docs/LIBERO_SOFT_EVAL.md) (progress-score eval).

---

## 1. Environment setup

The eval is **client–server**, which needs **two conda environments** (LIBERO pins old
`robosuite`/`mujoco`/`gym` that conflict with the training stack's torch-2.8 / py3.11; they talk over a
websocket):

- **`train`** — GPU env (py3.11) for **training** and **policy serving**.
- **`eval`** — CPU env (py3.10) for the **LIBERO MuJoCo sim client**.

A CUDA 12.x toolkit (`nvcc`, e.g. at `/usr/local/cuda`) is required even for eval (the policy server
imports `deepspeed`, which probes `nvcc` at import).

### 1a. `train` (training + serving, GPU, py3.11)

```bash
conda create -n train python=3.11 -y
conda activate train
export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH

# torch 2.8 (cu129) + this repo (editable)
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129

# flash-attn (optional but recommended; SDPA fallback works without it).
# Prebuilt wheel (fast) — match torch/py/abi; or `MAX_JOBS=32 pip install --no-build-isolation flash-attn` (~15 min compile):
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

pip install hf_transfer      # fast HF downloads
```

### 1b. `eval` (LIBERO sim client, CPU, py3.10)

```bash
conda create -n eval python=3.10 -y
conda activate eval
pip install "setuptools==65.5.0" "wheel==0.38.4" "pip==23.3.2"
pip install "numpy==1.24.4"
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install "robosuite==1.4.1" "mujoco==3.2.3" "bddl==1.0.1" "easydict==1.9" \
            "opencv-python==4.6.0.66" Pillow "matplotlib==3.5.3"
pip install "gym==0.25.2" --no-build-isolation

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$HOME/LIBERO"
pip install -e "$HOME/LIBERO" --no-deps
pip install websockets msgpack msgpack-numpy openpi-client tqdm tyro imageio imageio-ffmpeg \
            "hydra-core==1.2.0" termcolor future cloudpickle

# Point LIBERO at its bundled task files
mkdir -p ~/.libero && cat > ~/.libero/config.yaml <<YAML
benchmark_root: $HOME/LIBERO/libero/libero
bddl_files:     $HOME/LIBERO/libero/libero/bddl_files
init_states:    $HOME/LIBERO/libero/libero/init_files
datasets:       $HOME/LIBERO/libero/datasets
assets:         $HOME/LIBERO/libero/libero/assets
YAML
```

### 1c. Headless rendering libs + tokens

```bash
# Eval renders MuJoCo headless. OSMesa (CPU) is the robust default on busy multi-GPU nodes;
# EGL (GPU) is faster on a dedicated/idle GPU.
sudo apt-get update -y && sudo apt-get install -y libosmesa6 libgl1-mesa-glx libglfw3 libegl1 libgles2 libglvnd0
```

Create `./.env` (git-ignored — **never commit it**) with your tokens:

```bash
cat > .env <<'ENV'
HF_TOKEN=hf_...        # for downloading weights/dataset (+ checkpoint mirroring if used)
WANDB_API_KEY=...      # for training/eval logging (optional: WANDB_MODE=disabled to skip)
ENV
```

---

## 2. Download backbone weights + dataset

The experiments use the **Wan2.2-TI2V-5B** backbone (DiT + VAE + T5 encoder + umt5 tokenizer) plus the
**Wan2.1 CLIP** image encoder. These are loaded at model construction (the fine-tuned weights are an
overlay), so they are required for **both** training and eval.

```bash
conda activate train && set -a; . ./.env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p checkpoints data

# (a) Wan2.2-TI2V-5B backbone (~34 GB)
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
# (b) CLIP image encoder from Wan2.1 (~4.5 GB; Wan2.2 doesn't ship it)
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
# (c) umt5-xxl tokenizer (bundled inside Wan2.2 — copy it out)
mkdir -p ./checkpoints/umt5-xxl && cp ./checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl/* ./checkpoints/umt5-xxl/
```

**LIBERO dataset** — download then convert to the MP4 LeRobot format (the openpi LeRobot dataset stores
frames as PNG bytes inside parquet; this codebase reads on-disk MP4):

```bash
# (d) ~11 GB after conversion (1693 episodes / 273k frames)
hf download physical-intelligence/libero --repo-type dataset --local-dir ./data/libero_raw_lerobot
python scripts/data/convert_libero_to_dreamzero.py \
    --src data/libero_raw_lerobot --dst data/libero_lerobot --num-workers 64
```

---

## 3. Training

All launchers read the same env-var knobs and differ only in the action-head config they select.
Common knobs (with defaults):

| Env var | Default | Meaning |
|---|---|---|
| `NUM_GPUS` | 8 | data-parallel GPUs (ZeRO-2; **needs ≥ 2**) |
| `CUDA_VISIBLE_DEVICES` | (all) | which GPUs to use |
| `OUTPUT_DIR` | per-option default under `checkpoints/` | checkpoint dir (auto-resumes from latest `checkpoint-N`) |
| `LIBERO_DATA_ROOT` | `data/libero_lerobot` | converted dataset |
| `PER_DEVICE_BATCH_SIZE` | 1 | **bs=1 only** (grow the global batch via more GPUs / `gradient_accumulation_steps`) |
| `MAX_STEPS` | 100 | training steps |
| `SAVE_STEPS` / `SAVE_STRATEGY` | 500 / steps | checkpoint cadence |
| `TRAIN_ARCH` | full | `full` or `lora` |
| `PYTHON_BIN` | python | python used by `torch.distributed.run` |

`WAN22_CKPT_DIR`, `IMAGE_ENCODER_DIR`, `TOKENIZER_DIR` point at the §2 downloads (defaults assume
`./checkpoints/...`). Set a unique `WANDB_RUN_ID` per run (eval logs to a sibling `<id>-eval` run).

Pick the launcher for the option you want (here on GPUs 0–6, leaving GPU 7 for the eval watcher):

```bash
conda activate train && set -a; . ./.env; set +a
COMMON="NUM_GPUS=7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 LIBERO_DATA_ROOT=$PWD/data/libero_lerobot \
        PER_DEVICE_BATCH_SIZE=1 MAX_STEPS=100000 SAVE_STEPS=1000 SAVE_STRATEGY=steps TRAIN_ARCH=full \
        PYTHON_BIN=$(which python)"

# Option 1 — action_loss_only (baseline)
env $COMMON WANDB_RUN_ID=libero_action_loss_only \
    OUTPUT_DIR=$PWD/checkpoints/action_loss_only \
    bash scripts/train/libero_training_wan22_action_loss_only_skip_noisy_video.sh

# Option 2 — joint_action_loss_dynamics_loss
env $COMMON WANDB_RUN_ID=libero_joint \
    OUTPUT_DIR=$PWD/checkpoints/joint_action_loss_dynamics_loss \
    bash scripts/train/libero_training_wan22.sh

# Option 3 — decoupled_action_loss_dynamics_loss
env $COMMON WANDB_RUN_ID=libero_decoupled \
    OUTPUT_DIR=$PWD/checkpoints/decoupled_action_loss_dynamics_loss \
    bash scripts/train/libero_training_wan22_decoupled.sh

# Option 4 — use_generated_video_feedback
env $COMMON WANDB_RUN_ID=libero_feedback \
    OUTPUT_DIR=$PWD/checkpoints/use_generated_video_feedback \
    bash scripts/train/libero_training_wan22_future_shift.sh
```

> **Resume is automatic** from `OUTPUT_DIR` (no flag): a top-level `config.json` ⇒ "finished" (skips);
> else it resumes from the highest `checkpoint-N` (loading the DeepSpeed `global_step*` optimizer
> state — **`NUM_GPUS` must match the saved ZeRO rank count**); else it trains fresh. Use a new
> `OUTPUT_DIR` to start over.

---

## 4. Evaluation

Eval is **client–server**: a policy **server** (GPU, `train`) serves actions over a websocket; the
LIBERO **sim client** (CPU, `eval`) drives the simulator, scores success, and writes rollout MP4s + a
`metrics.json`. There are two ways to run it.

### 4a. Automatic watcher (recommended alongside training)

For every new `checkpoint-N` under `OUTPUT_DIR`, the watcher serves it on `SERVER_GPU`, runs the sim
eval, logs `eval/success_rate` to the sibling `<run>-eval` wandb run, keeps `latest-N ∪ best-N`
locally, and (optionally) mirrors best-N + latest (+ milestones) to a private HF repo.

```bash
# CONDA_SH/CONDA_ENV let the watcher self-activate conda. Point CONDA_ENV at your GPU/serving env
# (`train`); CONDA_SH defaults to /root/miniconda3.
OUTPUT_DIR=$PWD/checkpoints/action_loss_only SERVER_GPU=7 TRIALS=3 MAX_TASKS=3 \
    CONDA_ENV=train CONDA_SH=$(conda info --base)/etc/profile.d/conda.sh \
    bash scripts/eval/watch_eval_libero.sh --wandb-run-id libero_action_loss_only
```

Knobs (env): `SERVER_GPU` (7), `TRIALS` (per task), `MAX_TASKS` (3; `0`=all 10), `TASK_SUITE`
(`libero_spatial`), `KEEP_BEST`/`KEEP_LATEST`, `UPLOAD_REPO` (+`UPLOAD_BEST`, `MILESTONE_INTERVAL`,
`UPLOAD_MODEL_ONLY=1`), `MUJOCO_GL_BACKEND` (`osmesa`).

**`use_generated_video_feedback` (grounded feedback) is the same watcher with `CONTEXT_MODE=C REPLAN_STEPS=24`**
— optionally `SAVE_VIDEO_PRED=1` to dump the model's imagined video:

```bash
OUTPUT_DIR=$PWD/checkpoints/use_generated_video_feedback SERVER_GPU=7 TRIALS=3 MAX_TASKS=3 \
    CONTEXT_MODE=C REPLAN_STEPS=24 SAVE_VIDEO_PRED=1 \
    CONDA_ENV=train CONDA_SH=$(conda info --base)/etc/profile.d/conda.sh \
    bash scripts/eval/watch_eval_libero.sh --wandb-run-id libero_feedback-C
```

### 4b. Standalone eval (two terminals)

**Terminal A — policy server** (`train`, GPU). Point `--model_path` at a `checkpoint-N/` dir or a
finished top-level model. On Blackwell (`sm_103a`) prefix `TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1`
(the bundled `ptxas` can't target it); on Hopper/A100 it's optional.

```bash
conda activate train && set -a; . ./.env; set +a
CUDA_VISIBLE_DEVICES=0 python eval_utils/serve_dreamzero_libero.py \
    --model_path ./checkpoints/action_loss_only/checkpoint-20000 \
    --embodiment_tag libero_sim --tokenizer_path ./checkpoints/umt5-xxl --port 8000
```

**Terminal B — sim client** (`eval`, CPU render). Suites: `libero_spatial`,
`libero_object`, `libero_goal`, `libero_10`, `libero_90`.

```bash
conda activate eval
MUJOCO_GL=osmesa python eval_utils/run_libero_eval.py \
    --host 0.0.0.0 --port 8000 --task-suite-name libero_spatial \
    --num-trials-per-task 50 \
    --video-out-path ./eval_outputs/libero_spatial/videos \
    --metrics-out-path ./eval_outputs/libero_spatial/metrics.json
```

**`use_generated_video_feedback` with grounded feedback** — add `--context_mode C` to the server and
`--replan-steps 24` to the client (`future_frame_shift` is auto-detected from the checkpoint config):

```bash
# Terminal A (server)
CUDA_VISIBLE_DEVICES=0 python eval_utils/serve_dreamzero_libero.py \
    --model_path ./checkpoints/use_generated_video_feedback/checkpoint-20000 \
    --embodiment_tag libero_sim --tokenizer_path ./checkpoints/umt5-xxl --port 8000 \
    --context_mode C --save_video_pred --video_output_dir ./video_pred_output
# Terminal B (client)
MUJOCO_GL=osmesa python eval_utils/run_libero_eval.py \
    --host 0.0.0.0 --port 8000 --task-suite-name libero_spatial \
    --num-trials-per-task 50 --replan-steps 24 \
    --video-out-path ./eval_outputs/libero_spatial_C/videos
```

Useful client flags: `--max-tasks 1 --num-trials-per-task 2 --max-steps-override 60` (quick smoke),
`--progress-scores` (soft/partial-stage scores — see [`docs/LIBERO_SOFT_EVAL.md`](docs/LIBERO_SOFT_EVAL.md)),
`--no-save-videos`.

---

## 5. Quick smoke test (2 GPUs train, 1 GPU eval)

Validates that a chosen option trains and evals end-to-end on a tiny budget.

```bash
conda activate train && set -a; . ./.env; set +a

# Train: 2 GPUs, a couple of steps, save once.
NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 MAX_STEPS=4 SAVE_STEPS=2 SAVE_STRATEGY=steps \
    PER_DEVICE_BATCH_SIZE=1 TRAIN_ARCH=full PYTHON_BIN=$(which python) REPORT_TO=none \
    OUTPUT_DIR=$PWD/checkpoints/smoke_joint LIBERO_DATA_ROOT=$PWD/data/libero_lerobot \
    bash scripts/train/libero_training_wan22.sh

# Eval: 1 GPU server + tiny client run (1 task, 2 trials, short episodes).
CUDA_VISIBLE_DEVICES=2 python eval_utils/serve_dreamzero_libero.py \
    --model_path $PWD/checkpoints/smoke_joint/checkpoint-4 \
    --embodiment_tag libero_sim --tokenizer_path ./checkpoints/umt5-xxl --port 8000 &
# (in the eval env)
MUJOCO_GL=osmesa python eval_utils/run_libero_eval.py --host 0.0.0.0 --port 8000 \
    --task-suite-name libero_spatial --max-tasks 1 --num-trials-per-task 2 --max-steps-override 60 \
    --video-out-path ./eval_outputs/smoke/videos --metrics-out-path ./eval_outputs/smoke/metrics.json
```

---

## Notes / gotchas

- **`bs=1` only.** The action-head loss assumes one sample/device; grow the global batch with more
  GPUs and/or `training_args.gradient_accumulation_steps`.
- **Resume needs `NUM_GPUS` = saved ZeRO ranks.** A checkpoint saved with 7 ranks must resume with 7.
- **Eval rendering:** `MUJOCO_GL=osmesa` (CPU) is robust on busy nodes; `MUJOCO_GL=egl` is faster but
  aborts when its GPU is saturated. The watcher defaults to OSMesa.
- **Blackwell eval:** run the server eager (`TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1`); the
  bundled `ptxas` cannot target `sm_103a`. Training runs eager on all archs.
- **The Wan backbone is required at eval** (the fine-tuned weights are an overlay); only the training
  *dataset* is unnecessary for eval.
- **`.env` is git-ignored — never commit it.**
