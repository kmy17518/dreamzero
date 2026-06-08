# LIBERO Soft Eval — standalone progress-score evaluation (from an empty box)

Self-contained recipe to evaluate a **DreamZero‑LIBERO** checkpoint on a **single 80 GB GPU**, producing
both the usual **binary success rate** *and* the **soft / progress (partial-stage) scores** added in
`eval_utils/run_libero_eval.py`. Runs all **4 eval task suites** (`libero_spatial`, `libero_object`,
`libero_goal`, `libero_10`) with **5 trials/task** and writes **detailed per-task metrics**.

Assumes a fresh instance: **no conda env, no data, no checkpoints.** Eval needs **(1) the fine-tuned
checkpoint**, **(2) the umt5 tokenizer**, and **(3) the Wan backbone component weights** (DiT + T5
text encoder + VAE + CLIP image encoder). It does **not** need the training dataset.

> ⚠️ **The Wan backbone IS required at eval time** (this corrects an earlier version of this guide).
> The checkpoint's `config.json` carries **absolute** `*_pretrained_path` entries that point at
> `…/checkpoints/Wan2.2-TI2V-5B` and `…/checkpoints/Wan2.1-I2V-14B-480P`. At construction the action
> head **loads those backbone weights first**, then overlays the fine-tuned safetensors. The VAE / T5 /
> CLIP loaders fall back to an HF download if their local file is missing, but the **diffusion DiT loader
> has no fallback** — if `./checkpoints/Wan2.2-TI2V-5B/` exists (it will, after you drop the tokenizer
> there) it *requires* `diffusion_pytorch_model.safetensors.index.json` + shards to be present locally,
> or it raises `No safetensors file found …`. So §2 downloads the backbone explicitly. Because the repo
> is cloned at `$HOME/dreamzero` and the config paths are `…/dreamzero/checkpoints/…`, keep `$DZ`
> at `$HOME/dreamzero` (or edit `checkpoint-*/config.json` to match your path).

> What "soft eval" adds: each task's BDDL goal is decomposed into ordered stages scored every sim step
> from ground-truth state. Families: **pick-place** (`on`/`in` onto an object/container):
> `approach_src → grasp_src → approach_tgt → done`; **push** (floor-zone target / "push…" task):
> `approach_src → near_tgt → done`; **articulation** (`open`/`close`/`turnon`/`turnoff`):
> `approach → done`. Latching is strict-ordering + soft-credit, so the final stage equals the exact
> BDDL predicate (**progress = 1.0 ⇔ binary success**). See §4 for the metrics schema.

---

## 0. Prereqs

- **1 GPU with ≥ 80 GB** (any arch: H100/H200/A100-80G/B200/B300). Eval uses ~30–40 GB VRAM.
- **A CUDA 12.x toolkit with `nvcc` is REQUIRED**, plus `git`, `tmux`. This is **not optional**:
  the policy server imports `transformers` → `deepspeed`, which probes `nvcc` at import and **crashes
  with `MissingCUDAException: CUDA_HOME does not exist`** if there's no toolkit. A bare GPU instance
  often has only the *driver* (`nvidia-smi` works) but **no toolkit** (`nvcc` missing). Check, and if
  missing install one (any 12.x; major-version match with torch's cu12 is all that matters):
  ```bash
  ls /usr/local/cuda/bin/nvcc 2>/dev/null && /usr/local/cuda/bin/nvcc --version || echo "NO nvcc -> install a toolkit"
  # Easiest portable install (no sudo) — into the dreamzero env you create in §1a:
  #   conda activate dreamzero && conda install -y -c "nvidia/label/cuda-12.6.0" cuda-toolkit
  #   then use:  export CUDA_HOME=$CONDA_PREFIX
  # System-wide alternative (needs sudo; NVIDIA apt repo or a local cuda-repo-*):
  #   sudo apt-get install -y cuda-toolkit-12-6 && sudo ln -sfn /usr/local/cuda-12.6 /usr/local/cuda
  #   then use:  export CUDA_HOME=/usr/local/cuda
  ```
  Whichever you pick, **`export CUDA_HOME=…` must be set in the server shell (§3 Terminal A)**.
- **Headless MuJoCo rendering libs** (EGL = GPU rendering, the default below; OSMesa = CPU fallback):
  ```bash
  sudo apt-get update -y && sudo apt-get install -y \
      libosmesa6 libgl1-mesa-glx libglfw3 patchelf libegl1 libgles2 libglvnd0
  ```
  (The NVIDIA driver already ships `libEGL_nvidia`; the `libegl1`/`libglvnd0` loaders let MuJoCo's EGL
  backend find it. OSMesa stays installed as the robust fallback.)
- **Clone the repo and check out the `libero` branch** (where the soft-eval code lives). Pick any
  location; this guide uses `$HOME/dreamzero`:
  ```bash
  export DZ=$HOME/dreamzero
  git clone git@github.com:kmy17518/dreamzero.git "$DZ"   # or https://github.com/kmy17518/dreamzero.git
  cd "$DZ" && git checkout libero
  ```
- **`$DZ/.env`** (git-ignored) with a Hugging Face token that can read
  `kmy17518/dreamzero-libero-best` (**read scope is enough** — eval never uploads):
  ```
  HF_TOKEN=hf_...
  ```
- **GPU arch** (informational only — §3 now defaults to the **eager** server on every arch):
  ```bash
  python -c "import torch; print(torch.cuda.get_device_capability())" 2>/dev/null || true
  # (9,0)=Hopper / (8,0)=A100 / (10,x)=Blackwell
  ```
  We measured `torch.compile` to give **~0% speedup** here (eval is inference-bound on the diffusion
  forward, which compile doesn't accelerate for this shape) while adding a multi-minute warmup and
  mid-run recompile risk, so **eager is the default**. Blackwell must use eager anyway (bundled `ptxas`
  can't target `sm_103a`).

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
# CUDA_HOME = your toolkit from §0 (conda-env install -> $CONDA_PREFIX; system install -> /usr/local/cuda)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} && export PATH=$CUDA_HOME/bin:$PATH
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129   # torch 2.8 cu129

# flash-attn: there is NO matching prebuilt wheel on PyPI for torch2.8/cp311, so a plain
# `pip install flash-attn` falls back to a ~1-2h SOURCE COMPILE (and fails outright without nvcc).
# Install the matching prebuilt wheel DIRECTLY from the flash-attn GitHub releases instead:
python -m pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
python -m pip install hf_transfer         # fast HF downloads
```
> - **Do NOT** `pip install -U huggingface_hub` — `pip install -e .` pins a version with the `hf` CLI
>   *and* `permanently_delete_lfs_files`; upgrading breaks `transformers`/`tokenizers`.
> - **Match the flash-attn wheel** to your torch / python / C++-ABI. Check yours:
>   `python -c "import torch,sys; print(torch.__version__, torch._C._GLIBCXX_USE_CXX11_ABI, sys.version_info[:2])"`
>   (this env: torch `2.8.0+cu129`, cxx11abi `True`, py `3.11` → the `cu12torch2.8cxx11abiTRUE-cp311`
>   asset above). Browse https://github.com/Dao-AILab/flash-attention/releases for other combos.
> - **flash-attn is optional** — every import is guarded and the model has a PyTorch-SDPA fallback, so
>   eval still runs correctly (just slower) without it. The wheel is recommended.

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

## 2. Download what eval needs (checkpoint + tokenizer + Wan backbone)

```bash
conda activate dreamzero && cd "$DZ" && set -a; . ./.env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p checkpoints

# (a) Pick a checkpoint step from the mirror (e.g. the highest available, or a known-good one).
CKPT_STEP=$(python - <<'PY'
import os
from huggingface_hub import HfApi
a = HfApi(token=os.environ["HF_TOKEN"])
steps = {int(s.rfilename.split("/")[0].split("-")[1])
         for s in a.repo_info("kmy17518/dreamzero-libero-best", repo_type="model").siblings
         if s.rfilename.startswith("checkpoint-")}
print(max(steps))
PY
)
echo "Using checkpoint-$CKPT_STEP"

# (b) Download the MODEL ONLY (safetensors + experiment_cfg), skipping the huge DeepSpeed
#     global_step* optimizer state (not needed for eval). ~30 GB instead of ~130 GB.
hf download kmy17518/dreamzero-libero-best \
    --include "checkpoint-$CKPT_STEP/*" \
    --exclude "checkpoint-$CKPT_STEP/global_step*/*" \
    --local-dir ./checkpoints/dreamzero_libero_eval

# (c) Wan backbone components REQUIRED by the checkpoint's config.json (~28 GB), into the exact local
#     paths it references. NOTE: `hf download` only honors the LAST `--include` when repeated, so use
#     ONE `--include` per command (do NOT chain multiple --include flags).
WAN22=Wan-AI/Wan2.2-TI2V-5B; DST22=./checkpoints/Wan2.2-TI2V-5B
hf download "$WAN22" --include "diffusion_pytorch_model*"            --local-dir "$DST22"  # DiT shards + index
hf download "$WAN22" --include "Wan2.2_VAE.pth"                      --local-dir "$DST22"  # VAE
hf download "$WAN22" --include "models_t5_umt5-xxl-enc-bf16.pth"     --local-dir "$DST22"  # T5 text encoder
hf download "$WAN22" --include "config.json"                        --local-dir "$DST22"
hf download "$WAN22" --include "google/umt5-xxl/*"                  --local-dir "$DST22"  # umt5 tokenizer
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --local-dir ./checkpoints/Wan2.1-I2V-14B-480P                                         # CLIP image encoder

# (d) umt5-xxl tokenizer dir for --tokenizer_path in §3 (copy out of the backbone we just fetched).
mkdir -p ./checkpoints/umt5-xxl && cp ./checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl/* ./checkpoints/umt5-xxl/
```
> **Why the backbone is needed:** the fine-tuned safetensors are an *overlay*. At construction the action
> head loads the Wan2.2-TI2V-5B **DiT** (`diffusion_pytorch_model*`), **VAE** (`Wan2.2_VAE.pth`), **T5**
> (`models_t5_umt5-xxl-enc-bf16.pth`) and the Wan2.1 **CLIP** image encoder from the paths baked into
> `config.json`, *then* overlays the checkpoint. VAE/T5/CLIP auto-download to the HF cache if absent, but
> the **DiT loader requires the files locally** (no fallback) whenever `./checkpoints/Wan2.2-TI2V-5B/`
> already exists — which it does as soon as you put the tokenizer there. Downloading all of it locally
> (above) is the clean, deterministic path. (The training *dataset* is still **not** needed.)

---

## 3. Run the standalone soft eval (single GPU)

Two shells on the same box (use `tmux`): **Terminal A = policy server (`dreamzero`, GPU)**,
**Terminal B = sim client (`dreamzero_libero`, EGL/GPU rendering by default)**.

### Terminal A — policy server (GPU 0)
```bash
conda activate dreamzero && cd "$DZ" && set -a; . ./.env; set +a
# REQUIRED: point at the CUDA toolkit from §0 (else deepspeed import -> MissingCUDAException at startup).
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} && export PATH=$CUDA_HOME/bin:$PATH   # conda-env toolkit: CUDA_HOME=$CONDA_PREFIX
export CKPT_STEP=${CKPT_STEP:?set this to the step you downloaded, e.g. 36000}
export CKPT=$PWD/checkpoints/dreamzero_libero_eval/checkpoint-$CKPT_STEP

# EAGER server (default, every arch). torch.compile measured ~0% faster here (inference-bound) and adds
# warmup + recompile risk, so we keep eager. (To try compiled on Hopper/A100, drop the two vars below.)
TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 \
python eval_utils/serve_dreamzero_libero.py \
    --model_path "$CKPT" --embodiment_tag libero_sim \
    --tokenizer_path ./checkpoints/umt5-xxl --port 8000
```
Wait for `server listening on 0.0.0.0:8000` before starting Terminal B. (First start also fetches any
missing VAE/T5/CLIP into the HF cache; with §2 done it just loads the local files.)

### Terminal B — sim client: all 4 suites × 5 trials/task, with progress scores
```bash
conda activate dreamzero_libero && cd "$DZ"
export OUT=./eval_outputs/soft
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0      # GPU rendering, ~1.6x faster than osmesa (see Runtime note)
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo "==== $SUITE ===="
  for attempt in 1 2 3; do                       # auto-retry the suite if EGL ever SIGABRTs mid-run
    python eval_utils/run_libero_eval.py \
        --host 0.0.0.0 --port 8000 \
        --task-suite-name "$SUITE" \
        --num-trials-per-task 5 \
        --progress-scores \
        --video-out-path "$OUT/$SUITE/videos" \
        --metrics-out-path "$OUT/$SUITE/metrics.json" && break
    echo "[warn] $SUITE attempt $attempt failed; retrying in 10s..."; sleep 10
  done
done
```
If EGL fails to initialize on your box (rare), fall back to CPU rendering: `export MUJOCO_GL=osmesa`
and `unset MUJOCO_EGL_DEVICE_ID`.
- `--progress-scores` is **on by default**; thresholds `--approach-dist 0.07` and `--place-dist 0.12`
  (metres) are tunable but only affect the *partial-credit* numbers — `grasp_src` and `done` are exact.
- No `--max-tasks` ⇒ every task in the suite (10 each). Per-episode step caps are the suite defaults
  (`libero_spatial` 220, `libero_object` 280, `libero_goal` 300, `libero_10` 520).
- Metrics are written **incrementally** (a crash mid-run still leaves partial results).

**Smoke test first** (recommended, ~2 min) before the multi-hour full run:
```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python eval_utils/run_libero_eval.py --host 0.0.0.0 --port 8000 \
    --task-suite-name libero_goal --max-tasks 1 --num-trials-per-task 2 --max-steps-override 120 \
    --video-out-path ./eval_outputs/smoke/videos --metrics-out-path ./eval_outputs/smoke/metrics.json
```

> **Runtime (measured, A100-80G):** with **EGL** the bottleneck is policy **inference** (~1.6 s per
> replan call, issued every 5 sim-steps), *not* rendering — so EGL is **~1.6× faster end-to-end** than
> OSMesa (benchmarked: eager+OSMesa 140 s vs eager+EGL 86 s on a fixed 3-episode workload; `torch.compile`
> made no difference). Ballpark ~0.5–1.5 min/episode for the shorter suites, ~2–3 min for `libero_10`
> (520 steps). Full 4 suites × 10 tasks × 5 trials = **200 episodes ≈ ~3–4 h** (≈ **~2 h** at 3
> trials/task). OSMesa (`MUJOCO_GL=osmesa`) is the robust CPU fallback if EGL won't initialize.

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

1. **Eval needs checkpoint + tokenizer + Wan backbone** (see §2). The fine-tuned safetensors are an
   overlay on the Wan2.2-TI2V-5B DiT/VAE/T5 + Wan2.1 CLIP; the **DiT loader has no HF fallback** and
   raises `No safetensors file found …` if `./checkpoints/Wan2.2-TI2V-5B/` exists without the DiT shards.
   Only the training *dataset* is unnecessary.
2. **CUDA toolkit / `nvcc` is mandatory** (§0). Without it the server dies at import with
   `deepspeed … MissingCUDAException: CUDA_HOME does not exist`. A driver-only box (`nvidia-smi` works,
   `nvcc` missing) is the common trap. Install a 12.x toolkit and `export CUDA_HOME` in the server shell.
3. **flash-attn:** no PyPI wheel for torch2.8/cp311 → install the prebuilt wheel by URL (§1a), or skip it
   (guarded imports + SDPA fallback mean eval still runs, just slower). A plain `pip install flash-attn`
   triggers a long source build that fails without `nvcc`.
4. **`hf download` honors only the LAST `--include`** when the flag is repeated → use **one `--include`
   per command** (§2c). Silently downloads nothing for the dropped patterns otherwise.
5. **`--tokenizer_path` is mandatory on a fresh box.** The checkpoint's baked `tokenizer_path` is an
   absolute path from the training machine; pass `./checkpoints/umt5-xxl` to override it.
6. **Server = eager (default).** `torch.compile` measured ~0% here (inference-bound) with warmup +
   recompile risk. Blackwell must stay eager anyway (`ptxas` can't target `sm_103a`).
7. **Rendering = EGL (default), ~1.6× faster** than OSMesa (`export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0`).
   The doc's loop auto-retries a suite if EGL SIGABRTs; if EGL won't init at all, fall back to
   `MUJOCO_GL=osmesa`.
8. **Progress thresholds** (`--approach-dist`, `--place-dist`) are heuristics for the *intermediate*
   stages; `grasp_src` (fingerpad contact) and `done` (exact BDDL predicate) need no tuning. Because of
   soft-credit, `progress == 1.0` always coincides with binary success regardless of thresholds.
9. **Verify shard completeness.** Some steps on the mirror can be incomplete (e.g. `checkpoint-15000`
   was missing safetensors shards 1–2 and won't load). Confirm the local checkpoint has all
   `model-0000N-of-0000M.safetensors` referenced by `model.safetensors.index.json` before serving.
10. **Disk:** model-only checkpoint ≈ 30 GB; Wan backbone ≈ 28 GB; tokenizer a few MB; rollout MP4s a
    few hundred MB. Videos are saved by default — add **`--no-save-videos`** if you only want `metrics.json`.
11. **`.env` is git-ignored — never commit it.**
