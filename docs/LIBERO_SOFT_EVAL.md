# LIBERO Soft Eval (action-loss-only variant) — standalone progress-score evaluation (from an empty box)

Self-contained recipe to evaluate a **DreamZero‑LIBERO action-loss-only (+ skip-noisy-video)**
checkpoint on a **single 80 GB GPU**, producing both the usual **binary success rate** *and* the
**soft / progress (partial-stage) scores** added in `eval_utils/run_libero_eval.py`. Runs all
**4 eval task suites** (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`) with
**5 trials/task** and writes **detailed per-task metrics**.

Assumes a fresh instance: **no conda env, no data, no checkpoints.** Eval needs **(1) the checkpoint**,
**(2) the umt5 tokenizer**, and **HF access on the first serve** (to pull the Wan2.2/2.1 *encoder* base
weights — T5/VAE/CLIP, ~16 GB, cached once and reused across checkpoints). Those encoder weights are
immediately overwritten by the checkpoint's own weights, so **results are 100 % from the checkpoint**;
the **DiT is never downloaded** (it is filled straight from the checkpoint via `skip_component_loading`,
set in §2). **No training dataset.** The baked `*_pretrained_path` entries in `config.json` point at the
training box (`/root/...`) and are ignored on a fresh instance. The action-loss-only and skip-noisy-video
toggles are **baked into the checkpoint config** (`action_loss_only`, `action_skip_noisy_video`) and
reconstructed automatically at serve time — no extra serve flags needed.

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
- **`git`, `tmux`.** A **CUDA 12.x toolkit** at `/usr/local/cuda` (`nvcc`) is **optional** — only needed
  to *compile* flash-attn or to run the *compiled* server. The default recipe below (prebuilt flash-attn
  wheel + eager server) needs **no `nvcc`**, so a fresh box without a CUDA toolkit works as-is.
- **Headless MuJoCo rendering libs** — EGL (fast GPU rendering, the default in §3) **and** OSMesa
  (CPU fallback):
  ```bash
  sudo apt-get update -y && sudo apt-get install -y libegl1 libgl1-mesa-glx libglfw3 libosmesa6 patchelf
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
export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH   # only for the optional compile paths; harmless if /usr/local/cuda is absent
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129   # torch 2.8 cu129

# flash-attn: install a PREBUILT wheel matching this torch/python/ABI — no CUDA toolkit / nvcc, no
# ~15 min compile. (Plain `pip install flash-attn` tries to compile when it can't autodetect nvcc.)
python - <<'PY'
import torch, sys, subprocess
FA  = "2.8.3"                                           # release that ships cu12 + torch2.8 wheels
abi = "TRUE" if torch.compiled_with_cxx11_abi() else "FALSE"
py  = f"cp{sys.version_info.major}{sys.version_info.minor}"
url = (f"https://github.com/Dao-AILab/flash-attention/releases/download/v{FA}/"
       f"flash_attn-{FA}+cu12torch2.8cxx11abi{abi}-{py}-{py}-linux_x86_64.whl")
print("flash-attn wheel:", url)
subprocess.check_call([sys.executable, "-m", "pip", "install", url])
PY

# DeepSpeed is a TRAIN-only dep, but `transformers` imports it at serve time and DeepSpeed's importer
# calls nvcc/CUDA_HOME *just to import* — crashing the server on a box without a CUDA toolkit
# ("CUDA_HOME does not exist"). Eval never uses DeepSpeed, so remove it:
python -m pip uninstall -y deepspeed

python -m pip install hf_transfer         # fast HF downloads
```
> - **Do NOT** `pip install -U huggingface_hub` — `pip install -e .` pins a version with the `hf` CLI
>   *and* `permanently_delete_lfs_files`; upgrading breaks `transformers`/`tokenizers`.
> - **Have a CUDA toolkit and prefer to compile flash-attn?** `MAX_JOBS=96 python -m pip install
>   --no-build-isolation flash-attn` (Blackwell: set `TORCH_CUDA_ARCH_LIST="10.0+PTX"` first; Hopper
>   `"9.0+PTX"`). The prebuilt-wheel path above is simpler and needs no `nvcc`.

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

# (c) umt5-xxl tokenizer (a few MB) — bundled in the Wan2.2 repo at google/umt5-xxl. Download to a
#     SCRATCH dir, NOT ./checkpoints/Wan2.2-TI2V-5B: that exact name is the checkpoint's baked
#     diffusion_model_pretrained_path, and a tokenizer-only dir there makes serve crash with
#     "No safetensors file found at .../Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors".
hf download Wan-AI/Wan2.2-TI2V-5B --include "google/umt5-xxl/*" --local-dir ./checkpoints/_wan22_src
mkdir -p ./checkpoints/umt5-xxl && cp ./checkpoints/_wan22_src/google/umt5-xxl/* ./checkpoints/umt5-xxl/
rm -rf ./checkpoints/_wan22_src

# (d) Make the DiT load straight from the checkpoint: set skip_component_loading=true in every
#     downloaded checkpoint's config.json. Without it the server tries to (re)download the ~10 GB base
#     Wan2.2 DiT (and crashes if a partial ./checkpoints/Wan2.2-TI2V-5B exists). The 5B DiT is already in
#     the checkpoint safetensors, so this is loss-less and faster. (from_pretrained re-reads config.json
#     from disk and ignores CLI overrides, so patch the file.)
python - <<'PY'
import json, glob
for cfg in sorted(glob.glob("./checkpoints/dreamzero_libero_al_only_eval/checkpoint-*/config.json")):
    d = json.load(open(cfg))
    d["action_head_cfg"]["config"]["skip_component_loading"] = True
    json.dump(d, open(cfg, "w"), indent=2)
    print("patched", cfg)
PY
```
> **What serve loads:** the fine-tuned checkpoint's safetensors contain *all* final weights (DiT + T5 +
> VAE + CLIP + action head). With `skip_component_loading=true` (step d) the DiT is built empty and
> filled from the checkpoint — **no base-DiT download**. The T5/VAE/CLIP encoder *wrappers* still fetch
> their base weights from the public `Wan-AI/Wan2.2-TI2V-5B` / `Wan-AI/Wan2.1-I2V-14B-480P` repos on the
> **first** serve (~16 GB, cached in `~/.cache/huggingface`, reused across checkpoints), then are
> overwritten by the checkpoint's weights — so the served model is 100 % the checkpoint, but the first
> launch needs HF access. (To run fully offline, pre-download the full backbone and repoint the four
> `*_pretrained_path` entries — see `LIBERO_FINAL_INSTRUCTION.md` §3.) The action-loss-only /
> skip-noisy-video flags are read from the config (they only change how the action head attends — no
> extra weights), so the checkpoint serves with the right variant automatically.

---

## 3. Run the standalone soft eval (single GPU)

Two shells on the same box (use `tmux`): **Terminal A = policy server (`dreamzero`, GPU)**,
**Terminal B = sim client (`dreamzero_libero`, EGL GPU rendering by default)**.

### Terminal A — policy server (GPU 0)
```bash
conda activate dreamzero && cd "$DZ" && set -a; . ./.env; set +a
export CKPT_STEP=${CKPT_STEP:?set this to the step you downloaded, e.g. 36000}
export CKPT=$PWD/checkpoints/dreamzero_libero_al_only_eval/checkpoint-$CKPT_STEP

# EAGER server — the DEFAULT here: works on every GPU arch (incl. Blackwell) and avoids torch.compile
# surprises. First launch downloads the encoder base weights from HF (~16 GB, cached), so it can take a
# few minutes. (Optional: on Hopper/A100 drop the two TORCHDYNAMO/TORCH_COMPILE vars for a compiled
# server, but rendering+sim usually dominate, so the speedup is modest.)
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
# EGL = GPU rendering, the DEFAULT (much faster than OSMesa). If EGL ever SIGABRTs on your box, swap to
# the CPU fallback: MUJOCO_GL=osmesa (and drop MUJOCO_EGL_DEVICE_ID). Use 3 trials to run ~40% faster.
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo "==== $SUITE ===="
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python eval_utils/run_libero_eval.py \
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

**Smoke test first** (recommended, ~1–2 min) before the multi-hour full run:
```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python eval_utils/run_libero_eval.py --host 0.0.0.0 --port 8000 \
    --task-suite-name libero_goal --max-tasks 1 --num-trials-per-task 2 --max-steps-override 120 \
    --video-out-path ./eval_outputs/smoke/videos --metrics-out-path ./eval_outputs/smoke/metrics.json
```

> **Runtime:** with **EGL** GPU rendering, policy inference (~1.5 s/call, every 5 sim steps) dominates;
> figure ~0.5–1.5 min/episode (`libero_10` longer at 520 steps). The full 4 suites × 10 tasks × 5 trials
> = **200 episodes** ≈ 2–4 h (drop to **3 trials** to cut that ~40 %). If EGL SIGABRTs on a busy/shared
> GPU, fall back to `MUJOCO_GL=osmesa` (CPU, robust but ~2–3× slower). A compiled server (Hopper/A100,
> drop the dynamo vars) helps only modestly since rendering+sim, not inference, is the floor.

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

1. **Where weights come from.** The checkpoint safetensors hold *all* final weights. The DiT loads
   straight from the checkpoint (`skip_component_loading=true`, §2 step d). The T5/VAE/CLIP encoder
   *wrappers* fetch base weights from public HF (`Wan-AI/Wan2.2-TI2V-5B`, `Wan-AI/Wan2.1-I2V-14B-480P`)
   on the **first** serve (~16 GB, cached, reused), then are overwritten by the checkpoint — so the
   first launch needs HF access but the served model is 100 % the checkpoint. No training dataset.
2. **The variant is baked into the checkpoint.** `action_loss_only` and `action_skip_noisy_video` live in
   `checkpoint-*/config.json` (`action_skip_noisy_video` under `action_head_cfg.config.diffusion_model_cfg`)
   and are reconstructed at serve time, so the serve command is identical to any LIBERO checkpoint — no
   extra flags. Sanity-check the inference path offline (no GPU):
   `CUDA_VISIBLE_DEVICES="" ATTENTION_BACKEND=torch python scripts/test_action_skip_noisy_video.py`.
3. **No CUDA toolkit needed.** Use the **prebuilt flash-attn wheel** (§1a) and **uninstall deepspeed**
   (`transformers` imports it at serve time and its importer calls `nvcc`, crashing on a box without
   `/usr/local/cuda`). Both are done in §1a; eval needs neither `nvcc` nor DeepSpeed.
4. **Don't shadow the DiT path.** The checkpoint's baked `diffusion_model_pretrained_path` is literally
   `…/checkpoints/Wan2.2-TI2V-5B`; if that dir exists but lacks `diffusion_pytorch_model.safetensors`
   (e.g. a tokenizer-only download), serve crashes. §2 keeps the tokenizer out of that path **and** sets
   `skip_component_loading=true`, which sidesteps the DiT load entirely.
5. **`--tokenizer_path` is mandatory on a fresh box.** The checkpoint's baked `tokenizer_path` is an
   absolute training-machine path; pass `./checkpoints/umt5-xxl`.
6. **Server = eager (default).** `TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1` is universal. On
   Hopper/A100 you *may* drop both for a compiled server (modest gain; rendering/sim is the floor). On
   Blackwell keep eager (bundled `ptxas` can't target `sm_103a`).
7. **Rendering = EGL (default).** `MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0` is the fast GPU default; if it
   SIGABRTs on a busy/shared GPU, fall back to `MUJOCO_GL=osmesa` (CPU, robust, ~2–3× slower).
8. **Progress thresholds** (`--approach-dist`, `--place-dist`) are heuristics for the *intermediate*
   stages; `grasp_src` (fingerpad contact) and `done` (exact BDDL predicate) need no tuning. Because of
   soft-credit, `progress == 1.0` always coincides with binary success regardless of thresholds.
9. **Disk:** model-only checkpoint ≈ 24 GB; HF encoder cache ≈ 16 GB (one-time); tokenizer a few MB;
   rollout MP4s a few hundred MB. Add **`--no-save-videos`** if you only want `metrics.json`.
10. **`.env` is git-ignored — never commit it.**
