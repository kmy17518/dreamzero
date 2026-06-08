# LIBERO Soft Eval — action/dynamics-decoupled variant (from an empty box)

Standalone recipe to evaluate a **DreamZero‑LIBERO action/dynamics-decoupled** checkpoint on a
**single 80 GB GPU**, producing both the usual **binary success rate** *and* the **soft / progress
(partial-stage) scores** added in `eval_utils/run_libero_eval.py`. Runs all **4 eval task suites**
(`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`) with **5 trials/task** and writes
**detailed per-task metrics**.

This is the soft-eval guide for the **`libero_al_dl_decoupled`** branch, which carries the two
decoupled-attention variants from `docs/LIBERO.md`:
- **§12 — decoupled** (`decouple_action_dynamics`): action tokens do **not** attend to the
  generated video. Checkpoint dir `dreamzero_libero_wan22_decoupled`.
- **§13 — bidirectionally decoupled** (`decouple_action_dynamics` **and** `decouple_dynamics_action`):
  the mirror edge is also cut (video does not attend to action). Checkpoint dir
  `dreamzero_libero_wan22_decoupled_bidir`.

**Soft eval is variant-agnostic.** The progress scorer reads *ground-truth simulator state*
(object/gripper poses + LIBERO's own predicate/grasp helpers); it does not touch the policy, so the
exact same `run_libero_eval.py` works for the decoupled, bidir, action-loss-only, and joint/coupled
checkpoints. The decoupling itself is a **model-attention** change that is baked into the
checkpoint's `config.json` and rebuilt automatically when the policy server loads it — **no special
eval flag is required** (see §3).

Assumes a fresh instance: **no conda env, no data, no checkpoints.** Eval needs **(1) the decoupled
checkpoint**, **(2) the umt5 tokenizer**, and **(3) the Wan2.2-TI2V-5B + Wan2.1-I2V-14B-480P
backbones** — *no training dataset*. The backbone is required because the frozen T5/CLIP/VAE encoders
and the DiT base weights are loaded from `config.json`'s `*_pretrained_path` at model construction and
then overridden by the fine-tuned checkpoint (see §2); the loader uses the local backbone if present,
otherwise auto-downloads it from the public Wan repos.

> What "soft eval" adds: each task's BDDL goal is decomposed into ordered stages scored every sim step
> from ground-truth state. Families: **pick-place** (`on`/`in` onto an object/container):
> `approach_src → grasp_src → approach_tgt → done`; **push** (floor-zone target / "push…" task):
> `approach_src → near_tgt → done`; **articulation** (`open`/`close`/`turnon`/`turnoff`):
> `approach → done`. Latching is strict-ordering + soft-credit, so the final stage equals the exact
> BDDL predicate (**progress = 1.0 ⇔ binary success**). See §4 for the metrics schema.

---

## 0. Prereqs

- **1 GPU with ≥ 80 GB** (any arch: H100/H200/A100-80G/B200/B300). Eval uses ~30–40 GB VRAM.
- **CUDA 12.x toolkit** at `/usr/local/cuda` (provides `nvcc`), plus `git`, `tmux`.
- **Headless MuJoCo rendering libs** (OSMesa = robust CPU rendering on a single GPU):
  ```bash
  sudo apt-get update -y && sudo apt-get install -y libosmesa6 libgl1-mesa-glx libglfw3 patchelf
  ```
- **Clone the repo and check out the `libero_al_dl_decoupled` branch** (where the decoupled
  attention code *and* the soft-eval code both live). Pick any location; this guide uses
  `$HOME/dreamzero`:
  ```bash
  export DZ=$HOME/dreamzero
  git clone git@github.com:kmy17518/dreamzero.git "$DZ"   # or https://github.com/kmy17518/dreamzero.git
  cd "$DZ" && git checkout libero_al_dl_decoupled
  ```
  > **Why this branch (not `main`/`libero`) for the server too.** The decoupling is implemented in
  > `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`. Because §1a installs
  > `groot` **editable** (`pip install -e .`) *from this checkout*, the policy server imports the
  > decoupled attention and honors the `decouple_action_dynamics` / `decouple_dynamics_action` flags
  > baked into the checkpoint. Serving the same checkpoint from a checkout that lacks this code would
  > silently ignore the flags.
- **`$DZ/.env`** (git-ignored) with a Hugging Face token, only needed for **Option B** below
  (downloading a private checkpoint mirror); **read scope is enough** — eval never uploads:
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
cd "$DZ"                                  # IMPORTANT: install from the libero_al_dl_decoupled checkout
export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129   # torch 2.8 cu129
MAX_JOBS=96 python -m pip install --no-build-isolation flash-attn                     # usually a prebuilt wheel
python -m pip install hf_transfer         # fast HF downloads
```
> - **Install `groot` editable from `$DZ`** (this branch) so the server uses the decoupled attention.
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
> states) for all suites — that is everything the eval (and the progress scorer) needs. The
> `datasets/` entry can stay missing; you will see a harmless `datasets path ... does not exist`
> warning.

---

## 2. Get what eval needs (decoupled checkpoint + tokenizer + Wan backbone)

Pick **one** of the two checkpoint options, then get the tokenizer + Wan backbone (always required).

```bash
conda activate dreamzero && cd "$DZ"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p checkpoints

# Choose the variant you want to evaluate (decoupled = §12, bidir = §13).
export VARIANT=decoupled                 # or: decoupled_bidir
export CKPT_DIR=$PWD/checkpoints/dreamzero_libero_wan22_$VARIANT
```

### Option A — use a locally trained checkpoint (normal flow on this branch)
Train per `docs/LIBERO.md` §12.1 (decoupled) or §13.1 (bidir); checkpoints land under
`checkpoints/dreamzero_libero_wan22_<variant>/checkpoint-N/`. Nothing to download — just note a step:
```bash
export CKPT_STEP=36000                    # set to a checkpoint-N you actually have
ls "$CKPT_DIR/checkpoint-$CKPT_STEP"      # sanity: contains *.safetensors + config.json + experiment_cfg/
# Sanity-check it really is the decoupled variant. The flags are nested under
# action_head_cfg.config.diffusion_model_cfg in the saved config.json:
python - <<'PY'
import json, os
cfg = json.load(open(os.path.join(os.environ["CKPT_DIR"], f"checkpoint-{os.environ['CKPT_STEP']}", "config.json")))
dm = cfg["action_head_cfg"]["config"]["diffusion_model_cfg"]
print("decouple_action_dynamics =", dm.get("decouple_action_dynamics"))
print("decouple_dynamics_action =", dm.get("decouple_dynamics_action"))  # True only for the bidir variant
PY
```

### Option B — download a published mirror (from an empty box)
If you (or a teammate) mirrored the run to HF with the §12.2 / §13.2 eval-watcher
(`UPLOAD_REPO=<hf-user>/dreamzero-libero-<variant>-best`), pull the **model only** (skip the huge
DeepSpeed optimizer state):
```bash
set -a; . ./.env; set +a                  # HF_TOKEN (mirror repos are typically private)
export CKPT_REPO=kmy17518/dreamzero-libero-${VARIANT//_/-}-best   # e.g. .../dreamzero-libero-decoupled-best

# (a) Pick a checkpoint step from the mirror (e.g. the highest available, or a known-good one).
CKPT_STEP=$(python - <<'PY'
import os
from huggingface_hub import HfApi
a = HfApi(token=os.environ["HF_TOKEN"])
steps = {int(s.rfilename.split("/")[0].split("-")[1])
         for s in a.repo_info(os.environ["CKPT_REPO"], repo_type="model").siblings
         if s.rfilename.startswith("checkpoint-")}
print(max(steps))
PY
)
echo "Using $CKPT_REPO checkpoint-$CKPT_STEP"

# (b) Model only (safetensors + experiment_cfg), skipping global_step* optimizer state (~30 GB vs ~130 GB).
hf download "$CKPT_REPO" \
    --include "checkpoint-$CKPT_STEP/*" \
    --exclude "checkpoint-$CKPT_STEP/global_step*/*" \
    --local-dir "$CKPT_DIR"
export CKPT_STEP
```

### Tokenizer + Wan backbone (both options)
```bash
# (1) umt5-xxl tokenizer (a few MB) — bundled inside the Wan2.2 backbone repo at google/umt5-xxl.
#     The checkpoint's baked tokenizer_path is absolute and won't exist on a fresh box, so we pass
#     --tokenizer_path explicitly in §3.
hf download Wan-AI/Wan2.2-TI2V-5B --include "google/umt5-xxl/*" \
    --local-dir ./checkpoints/Wan2.2-TI2V-5B
mkdir -p ./checkpoints/umt5-xxl && cp ./checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl/* ./checkpoints/umt5-xxl/

# (2) Wan backbone weights — REQUIRED at eval (see note). Wan2.2-TI2V-5B provides the DiT, the T5
#     text encoder (models_t5_umt5-xxl-enc-bf16.pth), and the VAE (Wan2.2_VAE.pth); Wan2.1-I2V-14B-480P
#     provides the CLIP image encoder (models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth).
hf download Wan-AI/Wan2.2-TI2V-5B   --local-dir ./checkpoints/Wan2.2-TI2V-5B
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
```
> **Eval is NOT self-contained from the checkpoint — the Wan backbone is loaded at construction.**
> When the server builds the model it loads the **frozen** components (T5 text encoder, CLIP image
> encoder, VAE) *unconditionally* and the **DiT** backbone (unless `skip_component_loading`, which the
> LoRA-load path forces off) from the `*_pretrained_path` entries in `checkpoint-*/config.json`; the
> fine-tuned `checkpoint-*/*.safetensors` are then loaded **on top** to override the trained parts
> (you'll see `Missing keys ...` for the action-head-specific tensors — that's expected). Those frozen
> encoders/VAE are **not** stored in the checkpoint, so the backbone must be present.
> - The loader uses the **absolute** baked paths if they exist locally, **otherwise auto-downloads**
>   from the public `Wan-AI/Wan2.2-TI2V-5B` / `Wan-AI/Wan2.1-I2V-14B-480P` repos (no token needed).
>   A **locally trained** checkpoint (Option A) bakes absolute paths from the training box; pre-stage
>   the backbone at those paths (step (2) above puts them under `./checkpoints/...`, matching the
>   default training layout) or let the first server start download them.
> The decoupling flags (`decouple_action_dynamics` / `decouple_dynamics_action`) are likewise read from
> that `config.json` and applied when the server builds the DiT.

---

## 3. Run the standalone soft eval (single GPU)

Two shells on the same box (use `tmux`): **Terminal A = policy server (`dreamzero`, GPU)**,
**Terminal B = sim client (`dreamzero_libero`, CPU rendering)**. The example uses the **decoupled**
variant on port **8002**; for **bidir** use the `dreamzero_libero_wan22_decoupled_bidir` checkpoint
and port **8003** (the ports/dirs match `docs/LIBERO.md` §12.2 / §13.2 so concurrent evals don't
clash).

### Terminal A — policy server (GPU 0)
```bash
conda activate dreamzero && cd "$DZ"
export CKPT=$CKPT_DIR/checkpoint-${CKPT_STEP:?set this to the step you have, e.g. 36000}

# EAGER path (works on every GPU arch, incl. Blackwell). On Hopper/A100 you MAY drop the two
# TORCHDYNAMO/TORCH_COMPILE vars for a faster (compiled) server.
TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 \
python eval_utils/serve_dreamzero_libero.py \
    --model_path "$CKPT" --embodiment_tag libero_sim \
    --tokenizer_path ./checkpoints/umt5-xxl --port 8002
```
Wait for `server listening on 0.0.0.0:8002` before starting Terminal B. (No decouple flag is passed:
it is read from the checkpoint's `config.json` and applied because the server imports this branch's
decoupled attention code — see §0.)

### Terminal B — sim client: all 4 suites × 5 trials/task, with progress scores
```bash
conda activate dreamzero_libero && cd "$DZ"
export OUT=./eval_outputs/soft_decoupled        # bidir: ./eval_outputs/soft_decoupled_bidir
export PORT=8002                                 # bidir: 8003
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo "==== $SUITE ===="
  MUJOCO_GL=osmesa python eval_utils/run_libero_eval.py \
      --host 0.0.0.0 --port "$PORT" \
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
MUJOCO_GL=osmesa python eval_utils/run_libero_eval.py --host 0.0.0.0 --port "$PORT" \
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

One `metrics.json` per suite under `eval_outputs/soft_decoupled/<suite>/` (or
`…/soft_decoupled_bidir/<suite>/`). Top level keeps the original fields and adds
`overall_mean_progress`; each `per_task[]` gains a `progress` block. Example (pick-place task):

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
The soft progress is especially useful for the decoupled variants, where binary SR can be low: it
shows *how far* along each goal the action-decoupled policy got (e.g. reliably grasping but failing
the final place) rather than collapsing every near-miss to 0.

---

## 5. Notes / gotchas

1. **Eval needs checkpoint + tokenizer + Wan backbone** (no training dataset). The frozen T5/CLIP/VAE
   encoders and DiT base weights are loaded from `config.json`'s `*_pretrained_path` at construction and
   then overridden by the checkpoint, so `Wan2.2-TI2V-5B` and `Wan2.1-I2V-14B-480P` must be present
   locally (or are auto-downloaded from the public Wan repos on first start). See §2.
2. **Decoupling needs no eval flag, but the *server code* must be this branch.** The flags live in the
   checkpoint's `config.json`; install `groot` editable from the `libero_al_dl_decoupled` checkout
   (§1a) so the server rebuilds the gated attention. Verify with the Option-A `config.json` check
   (`decouple_action_dynamics=true`; plus `decouple_dynamics_action=true` for bidir).
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
7. **Ports / output dirs:** decoupled → port `8002`, `eval_outputs/soft_decoupled/`; bidir → port
   `8003`, `eval_outputs/soft_decoupled_bidir/`. Use distinct values from any concurrent joint-loss /
   action-loss-only eval so servers and videos don't collide.
8. **Disk:** model-only checkpoint ≈ 30 GB; tokenizer a few MB; rollout MP4s a few hundred MB. Videos
   are saved by default — add **`--no-save-videos`** to the client command if you only want
   `metrics.json`.
9. **`.env` is git-ignored — never commit it.**
