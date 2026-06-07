# DreamZero × LIBERO — run the current run from a fresh instance (compact)

This reproduces the run that is **live right now**: the **action-loss-only + skip-noisy-video**
variant of DreamZero on LIBERO (Wan2.2-TI2V-5B backbone, full fine-tune, bs=1), trained on **GPUs
0–6** with an **automatic eval+upload watcher on GPU 7**. For background/design see `LIBERO.md`
(§11 = action-loss-only, §12 = skip-noisy-video); this file is the self-contained "do this" recipe.

## 0. What this run is

| Item | Value |
|---|---|
| Branch | `libero_al_only` |
| Variant | action-loss-only (`train/loss == train/action_loss`) **+** `action_skip_noisy_video` (action attends to the clean video stream + state, **not** the noisy block being denoised) |
| Train launcher | `scripts/train/libero_training_wan22_action_loss_only_skip_noisy_video.sh` |
| Per-device batch | **1** (model limitation) |
| GPUs | train = 0–6 (`NUM_GPUS=7`), eval watcher = 7 |
| wandb | project `dreamzero_libero`, run `dreamzero_libero_wan22_action_loss_only_skip_noisy_video_v1` (eval → sibling `…_v1-eval`) |
| OUTPUT_DIR | `checkpoints/dreamzero_libero_wan22_action_loss_only_skip_noisy_video_v1` |
| HF mirror (private) | `kmy17518/dreamzero-libero-action-loss-only-skip-noisy-video-best-v1` = best-2 + latest + every 5k milestone (full, resumable checkpoints) |

Paths below assume the repo at `/root/dreamzero`, conda at `/root/miniconda3`, LIBERO at `/root/LIBERO`,
on an 8-GPU H100/H200 node.

---

## 1. Prereqs

- CUDA 12.x toolkit at `/usr/local/cuda` (provides `nvcc`), 8 GPUs, tmux.
- `apt-get install -y libosmesa6 libgl1-mesa-glx` (CPU rendering for eval).
- Create **`/root/dreamzero/.env`** (the repo does NOT auto-load it):
  ```bash
  cat > /root/dreamzero/.env <<'ENV'
  HF_TOKEN=hf_xxx                  # HF token with write access (downloads + checkpoint mirror)
  WANDB_API_KEY=xxx               # wandb logging
  ENV
  ```
- Get the code on the right branch:
  ```bash
  git clone <your dreamzero remote> /root/dreamzero && cd /root/dreamzero && git checkout libero_al_only
  ```

---

## 2. Conda + environments

```bash
# --- Miniconda ---
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /root/miniconda.sh
bash /root/miniconda.sh -b -p /root/miniconda3 && /root/miniconda3/bin/conda init bash
source /root/miniconda3/etc/profile.d/conda.sh

# --- Env 1: dreamzero (training + serving, GPU, py3.11) ---
conda create -n dreamzero python=3.11 -y -c conda-forge --override-channels
conda activate dreamzero
python -m ensurepip --upgrade                  # conda-forge python ships without pip
cd /root/dreamzero
export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129   # torch 2.8 cu129
MAX_JOBS=96 python -m pip install --no-build-isolation flash-attn                     # prebuilt wheel or ~15min compile
conda deactivate

# --- Env 2: dreamzero_libero (LIBERO MuJoCo sim client, CPU, py3.10) ---
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
conda deactivate
```

---

## 3. Base weights + dataset (one-time)

```bash
conda activate dreamzero && cd /root/dreamzero && set -a; . ./.env; set +a
python -m pip install -q -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p checkpoints data

# (a) Wan2.2-TI2V-5B backbone (~34 GB: DiT, VAE, T5 encoder, umt5 tokenizer)
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
# (b) CLIP image encoder from Wan2.1 (~4.5 GB; Wan2.2 doesn't ship it)
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
# (c) umt5-xxl tokenizer (bundled inside Wan2.2 — copy it out)
mkdir -p ./checkpoints/umt5-xxl && cp ./checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl/* ./checkpoints/umt5-xxl/

# (d) LIBERO dataset -> convert to DreamZero MP4 LeRobot format (~11 GB)
hf download physical-intelligence/libero --repo-type dataset --local-dir ./data/libero_raw_lerobot
python scripts/data/convert_libero_to_dreamzero.py \
    --src data/libero_raw_lerobot --dst data/libero_lerobot --num-workers 64
```

---

## 4. Train (+ eval watcher) — tmux, 7-GPU train + 1-GPU watcher

### 4a. (optional) Resume from the HF mirror
The HF repo holds **full** checkpoints (incl. the DeepSpeed `global_step*` optimizer), so training
resumes exactly. Download them into `OUTPUT_DIR`; training auto-resumes from the **highest** `checkpoint-N`.
Skip this block to **train from scratch**.

```bash
conda activate dreamzero && cd /root/dreamzero && set -a; . ./.env; set +a
export OUTPUT_DIR=$PWD/checkpoints/dreamzero_libero_wan22_action_loss_only_skip_noisy_video_v1
mkdir -p "$OUTPUT_DIR"
# everything the mirror has (best-2 + latest + milestones); resume uses the highest step:
hf download kmy17518/dreamzero-libero-action-loss-only-skip-noisy-video-best-v1 --local-dir "$OUTPUT_DIR"
# (or just the latest, e.g.: hf download <repo> --include "checkpoint-20000/*" --local-dir "$OUTPUT_DIR")
```

### 4b. Launch (two tmux panes)
```bash
tmux new-session -d -s libero -x 250 -y 50 && tmux split-window -h -t libero
```

**Pane 0 — training (GPUs 0–6, bs=1).** Run as ONE line:
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate dreamzero && cd /root/dreamzero && set -a; . ./.env; set +a
WANDB_RUN_ID=dreamzero_libero_wan22_action_loss_only_skip_noisy_video_v1 NUM_GPUS=7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 SAVE_TOTAL_LIMIT=100000 OUTPUT_DIR=$PWD/checkpoints/dreamzero_libero_wan22_action_loss_only_skip_noisy_video_v1 LIBERO_DATA_ROOT=$PWD/data/libero_lerobot PER_DEVICE_BATCH_SIZE=1 MAX_STEPS=100000 SAVE_STEPS=1000 SAVE_STRATEGY=steps TRAIN_ARCH=full PYTHON_BIN=$(which python) bash scripts/train/libero_training_wan22_action_loss_only_skip_noisy_video.sh
```

**Pane 1 — eval + upload watcher (GPU 7).** Run as ONE line:
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate dreamzero && cd /root/dreamzero
OUTPUT_DIR=$PWD/checkpoints/dreamzero_libero_wan22_action_loss_only_skip_noisy_video_v1 SERVER_GPU=7 TRIALS=3 KEEP_BEST=3 KEEP_LATEST=3 UPLOAD_BEST=2 MILESTONE_INTERVAL=5000 UPLOAD_REPO=kmy17518/dreamzero-libero-action-loss-only-skip-noisy-video-best-v1 bash scripts/eval/watch_eval_libero.sh --wandb-run-id dreamzero_libero_wan22_action_loss_only_skip_noisy_video_v1
```

Sanity: pane 0 prints `Using PYTHONPATH: /root/dreamzero` and the saved config shows **both**
`'action_loss_only': True` and `'action_skip_noisy_video': True`; pane 1 logs
`upload->… (best-2 + latest + milestones/5000, reclaim-on-evict)`.

What the watcher does each new `checkpoint-N` (GPU 7): serve it → run the LIBERO sim eval
(`libero_spatial`, first `MAX_TASKS=3` tasks × `TRIALS=3`, OSMesa) → log `eval/success_rate` to the
sibling wandb run → keep **latest-3 ∪ best-3** locally → mirror **best-2 + latest + every 5k
milestone** to the Hub, **permanently deleting** evicted LFS blobs so Hub storage stays bounded.

---

## 5. Standalone eval (no watcher) — all suites, all tasks

Client–server (two envs). Server = `dreamzero` (GPU); client = `dreamzero_libero` (CPU render).
First pick a checkpoint (a local `checkpoint-N/`, the finished top-level model dir, or download one
from the HF mirror):

```bash
conda activate dreamzero && cd /root/dreamzero && set -a; . ./.env; set +a
# e.g. grab a milestone (or latest) from the Hub:
hf download kmy17518/dreamzero-libero-action-loss-only-skip-noisy-video-best-v1 \
    --include "checkpoint-20000/*" --local-dir ./checkpoints/eval_ckpt
export EVAL_CKPT=./checkpoints/eval_ckpt/checkpoint-20000
```

**Terminal A — policy server (`dreamzero`):**
```bash
conda activate dreamzero && cd /root/dreamzero && set -a; . ./.env; set +a
CUDA_VISIBLE_DEVICES=0 python eval_utils/serve_dreamzero_libero.py \
    --model_path "$EVAL_CKPT" \
    --embodiment_tag libero_sim --tokenizer_path ./checkpoints/umt5-xxl --port 8000
```

**Terminal B — sim client (`dreamzero_libero`), all 5 suites, all tasks, default 50 trials/task:**
```bash
conda activate dreamzero_libero && cd /root/dreamzero
for suite in libero_spatial libero_object libero_goal libero_10 libero_90; do
  MUJOCO_GL=osmesa python eval_utils/run_libero_eval.py \
      --host 0.0.0.0 --port 8000 --task-suite-name "$suite" \
      --num-trials-per-task 50 \
      --video-out-path ./eval_outputs/$suite/videos \
      --metrics-out-path ./eval_outputs/$suite/metrics.json
done
```
Defaults: `--num-trials-per-task 50`, `--max-tasks 0` (= **all** tasks in the suite). Each suite
writes per-task + overall success to `eval_outputs/<suite>/metrics.json` plus per-episode rollout
MP4s (`…_success.mp4` / `…_failure.mp4`). One server serves all suites sequentially. Use
`MUJOCO_GL=egl` instead of `osmesa` only if a GPU is fully dedicated to rendering (faster, but EGL
aborts on a busy GPU — see `LIBERO.md` §10.7 #6).

---

## 6. Key facts / gotchas

- **Resume is automatic** from `OUTPUT_DIR`: if `OUTPUT_DIR/config.json` exists → "finished", skips;
  else resumes from the highest `checkpoint-N` (loads the `global_step*` optimizer); else trains fresh.
  Use a fresh `OUTPUT_DIR` to restart from scratch.
- **bs must be 1** (model limitation); grow the global batch with
  `training_args.gradient_accumulation_steps=N` and/or more GPUs.
- **Run the launcher as ONE line** — multi-line `\` paste into tmux can drop the inline `VAR=val` prefixes.
- **Single env (`dreamzero`) suffices to train + serve;** the LIBERO sim client needs `dreamzero_libero`.
- **HF storage:** the watcher mirrors best-2 + latest + 5k milestones and calls
  `permanently_delete_lfs_files` on eviction (HF bills LFS blobs across the *whole* commit history, so
  `delete_folder`/squash alone never free space). To reclaim manually:
  `HfApi().permanently_delete_lfs_files(repo, [f for f in HfApi().list_lfs_files(repo) if <not kept>])`.
- **Distinct names matter:** to launch a *new* parallel run, change `WANDB_RUN_ID`, `OUTPUT_DIR`, and
  `UPLOAD_REPO` (keep the wandb project `dreamzero_libero`); never reuse a deleted wandb run id.
- **Verify the attention variant offline (no GPU):**
  `CUDA_VISIBLE_DEVICES="" ATTENTION_BACKEND=torch python scripts/test_action_skip_noisy_video.py`.
- **On a busy/shared node**, keep training on GPUs 0–6 and the watcher's server on GPU 7
  (`SERVER_GPU=7`); the watcher renders on CPU (OSMesa) so it won't fight training for GPU memory.
```
