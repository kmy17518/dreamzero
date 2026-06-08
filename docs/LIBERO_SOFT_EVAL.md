# LIBERO Soft Eval (action-loss-only variant) — standalone progress-score evaluation (from an empty box)

Self-contained recipe to evaluate a **DreamZero‑LIBERO action-loss-only (+ skip-noisy-video)**
checkpoint on a **single 80 GB GPU**, producing both the usual **binary success rate** *and* the
**soft / progress (partial-stage) scores** added in `eval_utils/run_libero_eval.py`. Runs all
**4 eval task suites** (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`) with
**5 trials/task** and writes **detailed per-task metrics**.

Assumes a fresh instance: **no conda env, no data, no checkpoints.** Eval needs only **(1) the
checkpoint** and **(2) the umt5 tokenizer** — *no training dataset and no separate Wan backbone*
(the fine-tuned checkpoint safetensors are self-contained; the `*_pretrained_path` entries in its
`config.json` are vestigial and are **not** loaded at eval time). The action-loss-only and
skip-noisy-video toggles are **baked into the checkpoint config** (`action_loss_only`,
`action_skip_noisy_video`) and reconstructed automatically at serve time — the soft eval needs **no
extra flags** for this variant.

> What "soft eval" adds: each task's BDDL goal is decomposed into ordered stages scored every sim step
> from ground-truth state. Families: **pick-place** (`on`/`in` onto an object/container):
> `approach_src → grasp_src → approach_tgt → done`; **push** (floor-zone target / "push…" task):
> `approach_src → near_tgt → done`; **articulation** (`open`/`close`/`turnon`/`turnoff`):
> `approach → done`. Latching is strict-ordering + soft-credit, so the final stage equals the exact
> BDDL predicate (**progress = 1.0 ⇔ binary success**). The progress score is purely sim-side and is
> identical across model variants. See §4 for the metrics schema.

---

## 0. Prereqs

- **1 GPU with ≥ 80 GB** (any arch: H100/H200/A100-80G/B200/B300). Eval uses ~30–40 GB VRAM.
- **CUDA 12.x toolkit** at `/usr/local/cuda` (provides `nvcc`), plus `git`, `tmux`.
- **Headless MuJoCo rendering libs** (OSMesa = robust CPU rendering on a single GPU):
  ```bash
  sudo apt-get update -y && sudo apt-get install -y libosmesa6 libgl1-mesa-glx libglfw3 patchelf
  ```
- **Clone the repo and check out the `libero_al_only` branch** (where this variant + the soft-eval
  code live). Pick any location; this guide uses `$HOME/dreamzero`:
  ```bash
  export DZ=$HOME/dreamzero
  git clone git@github.com:kmy17518/dreamzero.git "$DZ"   # or https://github.com/kmy17518/dreamzero.git
  cd "$DZ" && git checkout libero_al_only
  ```
- **`$DZ/.env`** (git-ignored) with a Hugging Face token that can read
  `kmy17518/dreamzero-libero-action-loss-only-skip-noisy-video-best-v1` (**read scope is enough** —
  eval never uploads):
  ```
  HF_TOKEN=hf_...
  ```
- **Know your GPU arch** (decides the eval compile toggle in §3):
  ```bash
  python -c "import torch; print(torch.cuda.get_device_capability())" 2>/dev/null || true
  # (9,0)=Hopper / (8,0)=A100 -> torch.compile works (faster); (10,x)=Blackwell -> run EAGER (TORCHDYNAMO_DISABLE=1)
  ```
  When unsure, the **eager** path in §3 works on every arch.

---

## 1. Conda environments

Two envs: **`dreamzero`** (GPU policy server, py3.11) and **`dreamzero_libero`** (CPU MuJoCo sim
client, py3.10). If you have no conda:
```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3 && ~/miniconda3/bin/conda init bash && source ~/.bashrc
```

### 1a. Env `dreamzero` — policy server (GPU, py3.11)
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n dreamzero python=3.11 -y -c conda-forge --override-channels
conda activate dreamzero
python -m ensurepip --upgrade            # conda-forge python ships without pip
cd "$DZ"
export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129   # torch 2.8 cu129
MAX_JOBS=96 python -m pip install --no-build-isolation flash-attn                     # usually a prebuilt wheel
python -m pip install hf_transfer         # fast HF downloads
```
> - **Do NOT** `pip install -U huggingface_hub` — `pip install -e .` pins a version with the `hf` CLI
>   *and* `permanently_delete_lfs_files`; upgrading breaks `transformers`/`tokenizers`.
> - **flash-attn on Blackwell:** if the prebuilt wheel tries to compile, force the arch first, e.g.
>   `TORCH_CUDA_ARCH_LIST="10.0+PTX"` (Hopper: `"9.0+PTX"`).

### 1b. Env `dreamzero_libero` — LIBERO MuJoCo sim client (CPU, py3.10)
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n dreamzero_libero python=3.10 pip -y -c conda-forge --override-channels
conda activate dreamzero_libero
python -m pip install "setuptools==65.5.0" "wheel==0.38.4" "pip==23.3.2"
python -m pip install "numpy==1.24.4"
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "robosuite==1.4.1" "mujoco==3.2.3" "bddl==1.0.1" "easydict==1.9" \
            "opencv-python==4.6.0.66" Pillow "matplotlib==3.5.3"
python -m pip install "gym==0.25.2" --no-build-isolation
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$HOME/LIBERO"
python -m pip install -e "$HOME/LIBERO" --no-deps
python -m pip install websockets msgpack msgpack-numpy openpi-client tqdm tyro imageio \
            imageio-ffmpeg "hydra-core==1.2.0" termcolor future cloudpickle
# Point LIBERO at its bundled task files (these ship with the repo; eval resets use init_files/, NOT datasets/).
mkdir -p ~/.libero && cat > ~/.libero/config.yaml <<YAML
benchmark_root: $HOME/LIBERO/libero/libero
bddl_files:     $HOME/LIBERO/libero/libero/bddl_files
init_states:    $HOME/LIBERO/libero/libero/init_files
datasets:       $HOME/LIBERO/libero/datasets
assets:         $HOME/LIBERO/libero/libero/assets
YAML
```
> The LIBERO clone bundles `bddl_files/` (goal definitions) and `init_files/` (per-task initial
> states) for all suites — that is everything the eval needs. The `datasets/` entry can stay missing;
> you will see a harmless `datasets path ... does not exist` warning.

---

## 2. Download what eval needs (checkpoint + tokenizer only)

```bash
conda activate dreamzero && cd "$DZ" && set -a; . ./.env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
export CKPT_REPO=kmy17518/dreamzero-libero-action-loss-only-skip-noisy-video-best-v1
mkdir -p checkpoints

# (a) Pick a checkpoint step from the mirror (e.g. the highest available, or a known-good one).
CKPT_STEP=$(python - <<PY
import os
from huggingface_hub import HfApi
a = HfApi(token=os.environ["HF_TOKEN"])
steps = {int(s.rfilename.split("/")[0].split("-")[1])
         for s in a.repo_info(os.environ["CKPT_REPO"], repo_type="model").siblings
         if s.rfilename.startswith("checkpoint-")}
print(max(steps))
PY
)
echo "Using checkpoint-$CKPT_STEP"

# (b) Download the MODEL ONLY (safetensors + experiment_cfg), skipping the huge DeepSpeed
#     global_step* optimizer state (not needed for eval). ~30 GB instead of ~130 GB.
hf download "$CKPT_REPO" \
    --include "checkpoint-$CKPT_STEP/*" \
    --exclude "checkpoint-$CKPT_STEP/global_step*/*" \
    --local-dir ./checkpoints/dreamzero_libero_al_only_eval

# (c) umt5-xxl tokenizer (a few MB) — bundled inside the Wan2.2 backbone repo at google/umt5-xxl.
#     The checkpoint's baked tokenizer_path is absolute and won't exist on a fresh box, so we pass
#     --tokenizer_path explicitly in §3.
hf download Wan-AI/Wan2.2-TI2V-5B --include "google/umt5-xxl/*" \
    --local-dir ./checkpoints/Wan2.2-TI2V-5B
mkdir -p ./checkpoints/umt5-xxl && cp ./checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl/* ./checkpoints/umt5-xxl/
```
> **Why no backbone / dataset:** the fine-tuned checkpoint's safetensors contain *all* weights
> (DiT + T5 text encoder + VAE + CLIP). The `diffusion/text/vae/image *_pretrained_path` fields in
> `checkpoint-*/config.json` are only used when *training from the backbone*; at eval the modules are
> constructed empty and then fully populated from the safetensors, so those paths are never read.
> The action-loss-only / skip-noisy-video flags **are** read from the config (they only affect how
> the action head attends — no extra weights), so the same checkpoint serves with the right variant
> automatically. The only external artifact eval still needs is the **umt5 tokenizer** (to tokenize
> the prompt).

---

## 3. Run the standalone soft eval (single GPU)

Two shells on the same box (use `tmux`): **Terminal A = policy server (`dreamzero`, GPU)**,
**Terminal B = sim client (`dreamzero_libero`, CPU rendering)**.

### Terminal A — policy server (GPU 0)
```bash
conda activate dreamzero && cd "$DZ" && set -a; . ./.env; set +a
export CKPT_STEP=${CKPT_STEP:?set this to the step you downloaded, e.g. 36000}
export CKPT=$PWD/checkpoints/dreamzero_libero_al_only_eval/checkpoint-$CKPT_STEP

# EAGER path (works on every GPU arch, incl. Blackwell). On Hopper/A100 you MAY drop the two
# TORCHDYNAMO/TORCH_COMPILE vars for a faster (compiled) server.
TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 \
python eval_utils/serve_dreamzero_libero.py \
    --model_path "$CKPT" --embodiment_tag libero_sim \
    --tokenizer_path ./checkpoints/umt5-xxl --port 8000
```
Wait for `server listening on 0.0.0.0:8000` before starting Terminal B.

### Terminal B — sim client: all 4 suites × 5 trials/task, with progress scores
```bash
conda activate dreamzero_libero && cd "$DZ"
export OUT=./eval_outputs/soft
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo "==== $SUITE ===="
  MUJOCO_GL=osmesa python eval_utils/run_libero_eval.py \
      --host 0.0.0.0 --port 8000 \
      --task-suite-name "$SUITE" \
      --num-trials-per-task 5 \
      --progress-scores \
      --video-out-path "$OUT/$SUITE/videos" \
      --metrics-out-path "$OUT/$SUITE/metrics.json"
done
```
- `--progress-scores` is **on by default**; thresholds `--approach-dist 0.07` and `--place-dist 0.12`
  (metres) are tunable but only affect the *partial-credit* numbers — `grasp_src` and `done` are exact.
- No `--max-tasks` ⇒ every task in the suite (10 each). Per-episode step caps are the suite defaults
  (`libero_spatial` 220, `libero_object` 280, `libero_goal` 300, `libero_10` 520).
- Metrics are written **incrementally** (a crash mid-run still leaves partial results).

**Smoke test first** (recommended, ~2 min) before the multi-hour full run:
```bash
MUJOCO_GL=osmesa python eval_utils/run_libero_eval.py --host 0.0.0.0 --port 8000 \
    --task-suite-name libero_goal --max-tasks 1 --num-trials-per-task 2 --max-steps-override 120 \
    --video-out-path ./eval_outputs/smoke/videos --metrics-out-path ./eval_outputs/smoke/metrics.json
```

> **Runtime:** ~1–3 min/episode (OSMesa CPU rendering; `libero_10` is longer at 520 steps). The full
> 4 suites × 10 tasks × 5 trials = **200 episodes** ≈ a few hours. To speed up: use the **compiled**
> server on Hopper/A100 (drop the dynamo vars), and/or GPU rendering with `MUJOCO_GL=egl
> MUJOCO_EGL_DEVICE_ID=0` (faster, but on a single shared GPU EGL can occasionally SIGABRT — OSMesa is
> the robust default).

---

## 4. Output: `metrics.json` (binary success + soft progress)

One `metrics.json` per suite under `eval_outputs/soft/<suite>/`. Top level keeps the original fields
and adds `overall_mean_progress`; each `per_task[]` gains a `progress` block. Example (pick-place task):

```json
{
  "task_suite_name": "libero_spatial",
  "num_trials_per_task": 5,
  "overall_success_rate": 0.6,
  "overall_mean_progress": 0.83,
  "per_task": [
    {
      "task_id": 0,
      "task_description": "pick up the black bowl ... place it on the plate",
      "episodes": 5, "successes": 3, "success_rate": 0.6,
      "progress": {
        "mean_episode_progress": 0.85,
        "subgoals": [
          {
            "pred": "on", "args": ["akita_black_bowl_1", "plate_1"],
            "stages": ["approach_src", "grasp_src", "approach_tgt", "done"],
            "reach_fraction": [1.0, 1.0, 0.8, 0.6]
          }
        ],
        "episode_progress": [1.0, 1.0, 1.0, 0.75, 0.5],
        "episode_max_stages": [[4], [4], [4], [3], [2]]
      }
    }
  ]
}
```

How to read it:
- **`reach_fraction[k]`** = fraction of trials that reached at least stage *k* (so `reach_fraction[-1]`
  equals `success_rate`). Above: every trial approached + grasped the bowl, 80 % got it over the plate,
  60 % actually placed it.
- **`episode_progress`** = per-trial `max_stage_reached / num_stages` (soft-credit latched).
- **`mean_episode_progress`** / **`overall_mean_progress`** = those averaged per task / per suite.
- **Families differ in stage count:** push sub-goals have 3 stages
  (`approach_src → near_tgt → done`), articulation 2 (`approach → done`); multi-conjunct `libero_10`
  tasks list one entry per conjunct and the episode progress is their mean.

To get a single number across all 4 suites, average each suite's `overall_success_rate` (and
`overall_mean_progress`) — or pool `per_task[].progress.episode_progress` across the four files.

---

## 5. Notes / gotchas

1. **Eval is self-contained from the checkpoint** — only the checkpoint dir (model-only is fine) and the
   **umt5 tokenizer** are required. No training dataset, no separate Wan2.2/Wan2.1 backbone.
2. **The variant is baked into the checkpoint.** `action_loss_only` and `action_skip_noisy_video` are
   stored in `checkpoint-*/config.json` and applied automatically at serve time, so the eval commands
   here are identical to any other LIBERO checkpoint — no extra flags needed.
3. **`--tokenizer_path` is mandatory on a fresh box.** The checkpoint's baked `tokenizer_path` is an
   absolute path from the training machine; pass `./checkpoints/umt5-xxl` to override it.
4. **Compile toggle:** eager (`TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1`) is universal; on
   Hopper/A100 you can drop both for a faster compiled server. On Blackwell keep eager (bundled
   `ptxas` can't target `sm_103a`).
5. **Rendering backend:** `MUJOCO_GL=osmesa` (CPU) is the robust single-GPU default; `MUJOCO_GL=egl`
   (set `MUJOCO_EGL_DEVICE_ID=0`) is faster but can clash with the server on the same GPU.
6. **Progress thresholds** (`--approach-dist`, `--place-dist`) are heuristics for the *intermediate*
   stages; `grasp_src` (fingerpad contact) and `done` (exact BDDL predicate) need no tuning. Because of
   soft-credit, `progress == 1.0` always coincides with binary success regardless of thresholds.
7. **Disk:** model-only checkpoint ≈ 30 GB; tokenizer a few MB; rollout MP4s a few hundred MB. Videos
   are saved by default — add **`--no-save-videos`** to the client command if you only want
   `metrics.json`.
8. **`.env` is git-ignored — never commit it.**
