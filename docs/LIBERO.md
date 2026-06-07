# Training & Evaluating DreamZero on LIBERO (Wan2.2-TI2V-5B backbone, full fine-tune)

This guide documents everything needed to **train DreamZero on the LIBERO dataset using the
Wan2.2-TI2V-5B backbone (full fine-tune, no LoRA)** and to **evaluate it in the LIBERO MuJoCo
simulator** (auto-scoring success, saving rollout videos and a metrics JSON, like
`openpi/examples/libero/main.py`).

It records the critical decisions, the exact environment setup, the data conversion, the code
that was added/modified, and the commands used to verify training + eval.

> **Reading note.** The eval is **client–server**: a DreamZero policy server (GPU, `dreamzero`
> env) serves actions over a websocket; the LIBERO simulator runs as a separate client (CPU,
> `dreamzero_libero` env). This is why two conda environments are used.

---

## 0. TL;DR — what was built

New embodiment `libero_sim` (2 cameras, 8-dim eef state, 7-dim delta action) wired end-to-end:

| Area | File(s) |
|---|---|
| Embodiment projector index | `groot/vla/configs/model/dreamzero/transform/base.yaml` (`libero_sim: 33`) |
| Modality config + transform | `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` (`modality_config_libero_sim`, `transform_libero_sim`, registered in the `modality_configs`/`transforms`/`metadata_versions`/`fps` dicts) |
| Data config (Wan2.2) | `groot/vla/configs/data/dreamzero/libero_relative_wan22.yaml` |
| Language template + 2-view video layout | `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py` (`collate()` + `_prepare_video()` `LIBERO_SIM` branches) |
| Dataset conversion | `scripts/data/convert_libero_to_dreamzero.py` |
| Training launcher | `scripts/train/libero_training_wan22.sh` |
| Eval server (policy) | `eval_utils/serve_dreamzero_libero.py` |
| Eval client (sim) | `eval_utils/run_libero_eval.py` |

`EmbodimentTag.LIBERO_SIM = "libero_sim"` already existed in `groot/vla/data/schema/embodiment_tags.py`.

---

## 1. Hardware used & a critical constraint

- 8× H100 80GB, 208 CPUs, ~1.7 TB RAM, 22 TB disk.
- **GPU 0 was occupied by another user's job (~76 GB, 100% util) for the entire session.** All
  training/eval here therefore ran on **GPUs 1–7 (7 GPUs)**. The training launcher defaults to
  `NUM_GPUS=8`; to reproduce on a fully free node use `NUM_GPUS=8` and drop `CUDA_VISIBLE_DEVICES`.
  Per-GPU memory (and thus the optimal batch size) is activation-bound and essentially identical
  for 7 vs 8 GPUs with ZeRO-2.

---

## 2. Conda environments

Two environments are used (the eval is client–server, and LIBERO pins old
`robosuite`/`mujoco`/`gym` that conflict with DreamZero's torch 2.8 / py3.11 stack — a single
unified env is not practical, and is unnecessary because the two halves talk over a websocket).

Conda base used here: `/home/ubuntu/jiajun-stanford-lab/projects/miniconda3` (the only conda on
the box). Adjust if your conda differs.

### 2.1 `dreamzero` — training + policy serving (GPU)

```bash
conda create -n dreamzero python=3.11 -y
conda activate dreamzero

cd /home/ubuntu/minyeong/dreamzero
export CUDA_HOME=/usr/local/cuda                      # CUDA 12.8 toolkit present on the box
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
MAX_JOBS=32 pip install --no-build-isolation flash-attn
```

This installs torch 2.8 (cu129) and `flash-attn` 2.8.3 (compiles in ~15 min; needs `nvcc` + gcc).

### 2.2 `dreamzero_libero` — LIBERO MuJoCo simulator client (CPU)

The sim client needs **no GPU and no torch-CUDA** — it renders MuJoCo on CPU/EGL and talks to
the policy server over websocket. LIBERO's `benchmark` module imports `torch`, so a CPU torch is
installed.

```bash
conda create -n dreamzero_libero python=3.10 -y
conda activate dreamzero_libero

# pin build tooling first (gym 0.25.2 needs old setuptools/wheel)
pip install "setuptools==65.5.0" "wheel==0.38.4" "pip==23.3.2"
pip install "numpy==1.24.4"
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install "robosuite==1.4.1" "mujoco==3.2.3" "bddl==1.0.1" "easydict==1.9" \
            "opencv-python==4.6.0.66" Pillow "matplotlib==3.5.3"
pip install "gym==0.25.2" --no-build-isolation

# LIBERO itself (clone + editable, deps already installed above)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /home/ubuntu/minyeong/LIBERO
pip install -e /home/ubuntu/minyeong/LIBERO --no-deps

# client deps (DreamZero websocket protocol + misc)
pip install websockets msgpack msgpack-numpy openpi-client tqdm tyro imageio imageio-ffmpeg \
            "hydra-core==1.2.0" termcolor future cloudpickle
```

Create LIBERO's config (points at the bundled bddl/init/asset files):

```bash
mkdir -p ~/.libero && cat > ~/.libero/config.yaml <<'YAML'
benchmark_root: /home/ubuntu/minyeong/LIBERO/libero/libero
bddl_files:     /home/ubuntu/minyeong/LIBERO/libero/libero/bddl_files
init_states:    /home/ubuntu/minyeong/LIBERO/libero/libero/init_files
datasets:       /home/ubuntu/minyeong/LIBERO/libero/datasets
assets:         /home/ubuntu/minyeong/LIBERO/libero/libero/assets
YAML
```

Headless rendering: run the client with `MUJOCO_GL=egl` (use `MUJOCO_GL=glx` if you hit EGL
errors). The `datasets path ... does not exist` warning is harmless — we only need the simulator,
not LIBERO's HDF5 demos (the demos are already in our LeRobot dataset).

---

## 3. Weights / checkpoints

Set your token first (read from `/home/ubuntu/minyeong/.env`):

```bash
export HF_TOKEN=<your hf token>     # this repo reads it from ../.env
```

```bash
cd /home/ubuntu/minyeong/dreamzero && mkdir -p checkpoints data

# (a) Wan2.2-TI2V-5B backbone (~34 GB: DiT shards, Wan2.2_VAE.pth, T5 encoder, AND the umt5 tokenizer)
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B

# (b) CLIP image encoder — Wan2.2-TI2V-5B does NOT ship it; take it from Wan2.1 (one file, ~4.5 GB)
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --local-dir ./checkpoints/Wan2.1-I2V-14B-480P

# (c) umt5-xxl tokenizer files — bundled inside the Wan2.2 repo, just copy them out
mkdir -p ./checkpoints/umt5-xxl
cp ./checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl/* ./checkpoints/umt5-xxl/
```

**Decision:** we do **not** download the full `google/umt5-xxl` model (~50 GB). Only the tokenizer
files (`spiece.model`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`) are
needed, and Wan2.2-TI2V-5B already bundles them under `google/umt5-xxl/`. The T5 *encoder weights*
come from `Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth`.

### 3.1 The public DROID checkpoint (for the eval-harness test in §7.1)

```bash
# DreamZero-DROID public checkpoint (14B, Wan2.1) — skip the ~19 GB TensorRT engines
hf download GEAR-Dreams/DreamZero-DROID --local-dir ./checkpoints/DreamZero-DROID \
    --exclude "tensorrt/*"
```

> **Base weights for the DROID checkpoint are auto-downloaded.** The DROID checkpoint's frozen
> modules (Wan2.1 T5 encoder, Wan2.1 VAE `z_dim=16`, CLIP, and the base DiT) are loaded at model
> `__init__` from the paths baked into its config (`/mnt/amlfs-01/...`, which don't exist locally).
> The action head's `ensure_file()` therefore **auto-downloads the full `Wan-AI/Wan2.1-I2V-14B-480P`
> base (~68 GB) to the HF cache on first server start** (one-time; the checkpoint then overwrites
> these frozen weights). This is expected and only needed for the harness test, not for the LIBERO
> model. To pre-cache it, run `hf download Wan-AI/Wan2.1-I2V-14B-480P` beforehand.

---

## 4. LIBERO dataset: download + convert

### 4.1 The format problem (critical)

openpi's `physical-intelligence/libero` LeRobot dataset stores camera frames **inside the parquet
files** as PNG bytes (`info.json` features `image`/`wrist_image` have `dtype: "image"`,
`total_videos: 0`). DreamZero's loader (`ShardedLeRobotSubLangSingleActionChunkDatasetDROID`)
reads frames from **on-disk MP4** under `videos/` via decord. So the dataset must be converted.

### 4.2 Download + convert

```bash
export HF_TOKEN=<your hf token>
hf download physical-intelligence/libero --repo-type dataset --local-dir ./data/libero_raw_lerobot

# Re-encode images -> MP4, rewrite parquet without image columns, write GEAR meta/ files.
python scripts/data/convert_libero_to_dreamzero.py \
    --src data/libero_raw_lerobot \
    --dst data/libero_lerobot \
    --num-workers 64
```

`convert_libero_to_dreamzero.py` (added here):
1. Decodes `image`/`wrist_image` and writes `videos/chunk-XXX/observation.images.{image,wrist_image}/episode_*.mp4` at the dataset fps (10), H.264 / yuv420p, exactly one frame per parquet row.
2. Rewrites each parquet keeping `state, actions, timestamp, frame_index, episode_index, index, task_index` (drops the image columns).
3. Writes `meta/info.json` with the two cameras as `dtype: "video"`.
4. Writes `meta/modality.json` for `libero_sim` (see below).
5. Computes `meta/stats.json` with **mean/std/min/max/q01/q99** for `state`+`actions` (the openpi stats only had mean/std/min/max — DreamZero's `q99` normalization needs `q99`/`q01`).
6. Writes `meta/embodiment.json` and copies `tasks.jsonl` / `episodes.jsonl`.

Result: `data/libero_lerobot/` (~11 GB), 1693 episodes, 273k frames, 28 shards.

### 4.3 `modality.json` mapping (authored by the converter)

```
state  (8-dim "state" column):  eef_position[0:3], eef_rotation[3:6] (axis-angle), gripper_state[6:8]
action (7-dim "actions" column): eef_delta[0:6],   gripper_action[6:7]
video:  image -> observation.images.image,  wrist_image -> observation.images.wrist_image
annotation: task -> task_index   (resolved to text via tasks.jsonl)
```

**Decision — `relative_action: false`.** LIBERO uses robosuite's `OSC_POSE` controller
(`LIBERO/libero/libero/envs/env_wrapper.py:17`), so the recorded `actions` are *already* delta
end-effector commands (6 delta-pose + 1 gripper). The model therefore predicts them directly; there
is no "subtract current state" conversion (unlike DROID, which uses `relative_action: true` on
absolute joint positions). At eval the predicted deltas are sent straight to the environment.

Source / openpi parity: this comes from openpi's own LIBERO config —
`openpi/src/openpi/training/config.py` `LeRobotLiberoDataConfig.create()` comments state *"In
Libero, the raw actions in the dataset are already delta actions, so we do not need to apply a
separate delta conversion"*, and `openpi/src/openpi/policies/libero_policy.py` passes `actions`
through unchanged. **openpi is not uniform, though:** its flagship `pi05_libero` config uses
`extra_delta_transform=False` (no conversion — same as us), while the older `pi0_libero`,
`pi0_libero_low_mem_finetune`, `pi0_fast_libero`, and `pi0_fast_libero_low_mem_finetune` configs
set `extra_delta_transform=True`, which applies `DeltaActions(make_bool_mask(6, -1))` — i.e. it
*additionally* subtracts the current proprio state from the first 6 action dims (gripper left
absolute), kept for compatibility with old Pi0 base checkpoints. DreamZero's `relative_action: true`
is the analog of that `DeltaActions` step; we follow the cleaner **pi05** convention
(`relative_action: false`). (With our `modality.json` key names — action `eef_delta` vs state
`eef_position`/`eef_rotation` — `relative_action: true` wouldn't even find a matching state sub-key,
so `false` is also the only consistent choice here.)

---

## 5. Code changes (the new `libero_sim` embodiment)

1. **`base.yaml`**: added `libero_sim: 33` to `embodiment_tag_to_projector_index`. (Needed so
   `collate()` / `DreamTransform` don't crash on the new embodiment. Note: `CausalWanModel`
   hardcodes `embodiment_id=0` internally, so the index value only matters for the collate
   language-template branch, not for separate projector weights.)
2. **`base_48_wan_fine_aug_relative.yaml`**: added `modality_config_libero_sim` (2 video keys, 3
   state sub-keys, 2 action sub-keys, `annotation.task`) + `transform_libero_sim` (q99 norm), and
   registered `libero_sim` in `modality_configs`, `transforms`, `metadata_versions`, `fps`.
3. **`libero_relative_wan22.yaml`**: new data config (160×320, `relative_action: false`,
   `libero_data_root`, mixture spec keyed by `libero_sim`).
4. **`dreamzero_cotrain.py`**:
   - `collate()`: added a `LIBERO_SIM` language-template branch ("…split into two views: left =
     agent camera, right = wrist…").
   - `_prepare_video()`: added a `LIBERO_SIM` branch that places the 2 views **side by side**
     `[agentview | wrist]` → `(H, 2W)`. (The action head then resizes the composite to 160×320.)

`num_views=2` is passed on the training CLI.

---

## 6. Training (Wan2.2-5B, full fine-tune, no LoRA)

```bash
conda activate dreamzero
cd /home/ubuntu/minyeong/dreamzero
export HF_TOKEN=<your hf token>
export WANDB_MODE=disabled        # or set up wandb and use REPORT_TO=wandb

NUM_GPUS=8 \
LIBERO_DATA_ROOT=$PWD/data/libero_lerobot \
OUTPUT_DIR=$PWD/checkpoints/dreamzero_libero_wan22 \
PER_DEVICE_BATCH_SIZE=1 \
MAX_STEPS=30000 \
SAVE_STEPS=1000 SAVE_STRATEGY=steps \
TRAIN_ARCH=full \
PYTHON_BIN=$(which python) \
bash scripts/train/libero_training_wan22.sh
```

> Keep `PER_DEVICE_BATCH_SIZE=1` (see §6.1 — the model only supports per-device batch 1). To grow
> the global batch, add `training_args.gradient_accumulation_steps=N` and/or use more GPUs.

Key config (set by the launcher): `data=dreamzero/libero_relative_wan22`,
`model/dreamzero/action_head=wan_flow_matching_action_tf_wan22`, 160×320 video (latent 10×20,
`frame_seqlen=50`), `num_frames=33`, `action_horizon=24`, `num_frame_per_block=2`,
`num_action_per_block=24`, `num_views=2`, DeepSpeed ZeRO-2 (`groot/vla/configs/deepspeed/zero2.json`),
`train_architecture=full`, `save_lora_only=false`.

The frozen modules load from local files (no downloads): DiT base = `Wan2.2-TI2V-5B`,
`text_encoder_pretrained_path=…/models_t5_umt5-xxl-enc-bf16.pth`,
`image_encoder_pretrained_path=…/Wan2.1…/models_clip…pth`,
`vae_pretrained_path=…/Wan2.2_VAE.pth`, `tokenizer_path=…/umt5-xxl`.

### 6.1 Optimal batch size (measured)

Full fine-tune of the 5B model with ZeRO-2 on H100 80GB:

| `per_device_train_batch_size` | result |
|---|---|
| **1** | **works; ~46.5 GB / GPU, ~1.7 s/step** ✅ (this is the optimal/maximum) |
| 2 | **crashes** in the action-head loss: `RuntimeError: The size of tensor a (2) must match the size of tensor b (96)` at `wan_flow_matching_action_tf.py:795` ❌ |

**Conclusion: the optimal (and maximum supported) per-device batch size is `1`.** This is a model
limitation, not a memory limit — bs=1 uses only ~46.5 GB of 80 GB, but the action-head training
`forward` (the joint video+action flow-matching loss, e.g. `has_real_action[:, None] * action_loss_per_sample`
and the action-register packing) assumes one sample per device. **Every official DreamZero training
script (`droid_training*.sh`, `agibot_training.sh`, `yam_training.sh`) uses
`per_device_train_batch_size=1`** for the same reason.

To grow the **effective/global** batch size, scale the orthogonal knobs instead:
- more data-parallel GPUs (global batch = `NUM_GPUS × 1 × grad_accum`), and/or
- `gradient_accumulation_steps` (HF Trainer; calls the bs=1 forward N times) — e.g. add
  `training_args.gradient_accumulation_steps=4` to the launcher overrides.

Training step time ≈ 1.7 s/step (bs=1, 7 GPUs); the only slow gap is the ~20 s shard cache when
the sharded loader moves to a new shard. `loss_log.jsonl` records `dynamics_loss` (video) and
`action_loss`.

### 6.2 Resume (verified)

Resume is **automatic from `OUTPUT_DIR`** — there is no resume flag:
- If `OUTPUT_DIR/config.json` exists → training is considered **finished** → the run exits.
- Else if `OUTPUT_DIR/checkpoint-*` exist → it resumes from the latest (`Resuming training from …/checkpoint-N`, loading the DeepSpeed `global_step*` optimizer state).

Verified here: a run was interrupted after `checkpoint-3`; rerunning the same command resumed from
step 3, ran to step 6, saved `checkpoint-6`, and wrote the final model. To restart from scratch,
use a fresh `OUTPUT_DIR` (or delete the old one).

---

## 7. Evaluation in the LIBERO simulator

Two processes. **Terminal A** (GPU, `dreamzero`) runs the policy server; **Terminal B** (CPU,
`dreamzero_libero`) runs the simulator client.

### 7.0 Real eval of a LIBERO-trained model

Terminal A — server:
```bash
conda activate dreamzero
cd /home/ubuntu/minyeong/dreamzero
CUDA_VISIBLE_DEVICES=1 python eval_utils/serve_dreamzero_libero.py \
    --model_path ./checkpoints/dreamzero_libero_wan22 \
    --embodiment_tag libero_sim \
    --tokenizer_path ./checkpoints/umt5-xxl \
    --port 8000
```

Terminal B — client (sim):
```bash
conda activate dreamzero_libero
cd /home/ubuntu/minyeong/dreamzero
MUJOCO_GL=egl python eval_utils/run_libero_eval.py \
    --host 0.0.0.0 --port 8000 \
    --task-suite-name libero_spatial \
    --num-trials-per-task 50 \
    --video-out-path ./eval_outputs/libero_spatial/videos
```

Outputs (like `openpi/examples/libero/main.py`, plus a metrics file):
- per-episode rollout MP4s under `--video-out-path` (named `…_success.mp4` / `…_failure.mp4`),
- `eval_outputs/libero_spatial/metrics.json` with per-task and overall success rates (written
  incrementally so a crash leaves partial results).

Task suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`.

### 7.1 Eval-harness smoke test with the public DreamZero-DROID checkpoint (verified)

This proves the eval pipeline end-to-end **before** a LIBERO model is trained, using the public
14B DROID checkpoint as a stand-in. The DROID checkpoint is a **different embodiment** (3 views,
joint-position actions), so the server adapts LIBERO obs to the DROID format (agentview duplicated
into the 2 exterior views; 8-dim state mapped into joint(7)+gripper(1)). **Actions are therefore
meaningless for LIBERO → success ≈ 0**; the point is to verify connection → obs → action →
sim → video/metrics.

Terminal A — server (`--embodiment_tag oxe_droid`):
```bash
conda activate dreamzero
cd /home/ubuntu/minyeong/dreamzero
CUDA_VISIBLE_DEVICES=1 python eval_utils/serve_dreamzero_libero.py \
    --model_path ./checkpoints/DreamZero-DROID \
    --embodiment_tag oxe_droid \
    --tokenizer_path ./checkpoints/umt5-xxl \
    --port 8000
```
(On first start this auto-downloads the Wan2.1 base ~68 GB to the HF cache — see §3.1 — and the
14B model load takes a few minutes.)

Terminal B — client:
```bash
conda activate dreamzero_libero
cd /home/ubuntu/minyeong/dreamzero
MUJOCO_GL=egl python eval_utils/run_libero_eval.py \
    --host 0.0.0.0 --port 8000 --task-suite-name libero_spatial \
    --max-tasks 1 --num-trials-per-task 2 --max-steps-override 60 \
    --video-out-path ./eval_outputs/harness_droid/videos
```
Verified result: 2 episodes ran, rollout MP4s + `metrics.json` written, overall success 0.0
(expected for the wrong embodiment). First inference on the 14B model warms up for a few minutes;
subsequent ones are ~3 s.

### 7.2 Eval observation details (must match training)

The client mirrors `openpi/examples/libero/main.py`:
- **180° rotation** of both camera images (`obs[...][::-1, ::-1]`) to match how the LIBERO demos
  (and hence our LeRobot dataset) are oriented.
- 8-dim state = `concat(eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2))`.
- `replan_steps=5` (execute 5 of the returned action chunk before re-querying).
- A per-episode `session_id`; on a new `session_id` the server resets the action head's causal
  KV-cache (`current_start_frame=0`).

Protocol note: this uses **DreamZero's** websocket protocol (`eval_utils/policy_client.py`, which
sends an `endpoint` field and supports `reset`), **not** openpi's `websocket_client_policy`
(openpi's server has no `endpoint`/`reset`). The server returns `{"actions": (N, 7)}`.

---

## 8. Critical decisions / gotchas (summary)

1. **Two envs, by design.** Eval is client–server; the LIBERO sim env (old robosuite/mujoco/gym,
   py3.10) is kept separate from the training/serving env (torch 2.8 / py3.11). A unified env is
   not practical and not needed.
2. **LIBERO data must be re-encoded to MP4.** `physical-intelligence/libero` stores images in
   parquet; DreamZero reads MP4. Use `convert_libero_to_dreamzero.py`.
3. **`relative_action: false` for LIBERO** (actions are already deltas).
4. **2 views, side-by-side** composite; `num_views=2`; new `collate()` + `_prepare_video()`
   branches for `LIBERO_SIM`.
5. **Stats need q01/q99**; the converter recomputes them (openpi's stats lacked them).
6. **`libero_sim: 33`** added to the projector-index map (only used by the collate language
   template; the DiT forces `embodiment_id=0`).
7. **umt5 tokenizer is bundled in Wan2.2-TI2V-5B**; no separate 50 GB download.
8. **Resume is automatic from `OUTPUT_DIR`**; a top-level `config.json` means "finished" (use a
   fresh dir to retrain).
9. **GPU 0 was busy** during this work → trained/evaled on GPUs 1–7. Use `NUM_GPUS=8` on a free
   node.
10. **DROID-checkpoint eval is a harness test only** (wrong embodiment → ~0 success).

---

## 9. Reproduction checklist

```text
[ ] conda env `dreamzero` (py3.11, pip install -e . cu129, flash-attn)           # §2.1
[ ] conda env `dreamzero_libero` (py3.10, robosuite/mujoco/gym + LIBERO + client) # §2.2
[ ] ~/.libero/config.yaml                                                          # §2.2
[ ] download Wan2.2-TI2V-5B, Wan2.1 CLIP, copy umt5 tokenizer                      # §3
[ ] (for harness test) download DreamZero-DROID + Wan2.1 T5/VAE                    # §3.1
[ ] hf download physical-intelligence/libero (dataset)                            # §4.2
[ ] python scripts/data/convert_libero_to_dreamzero.py                            # §4.2
[ ] bash scripts/train/libero_training_wan22.sh  (TRAIN_ARCH=full)               # §6
[ ] verify resume (rerun same OUTPUT_DIR)                                         # §6.2
[ ] serve_dreamzero_libero.py + run_libero_eval.py                               # §7
```

---

# 10. Update (latest iteration): per-device bs>1, eval/upload watcher, and a verified fresh-instance recipe

> This section is **self-contained and supersedes the paths in §1–§9** (which used
> `/home/ubuntu/minyeong/...`). On the current instance everything lives under `/root`:
> repo `/root/dreamzero`, conda `/root/miniconda3`, LIBERO `/root/LIBERO`, on **8× H200 (143 GB)**.
> Adjust paths/GPU counts for your box. Secrets live in `/root/dreamzero/.env`
> (`HF_TOKEN=...` and `WANDB_API_KEY=...`); the repo does **not** auto-load `.env`, so every shell
> must `set -a; . ./.env; set +a` before training/eval/uploads.

## 10.0 What changed in this iteration

- **Per-device batch size > 1 (branch `libero_distributed`).** Two fixes are required and both live
  on `libero_distributed` (the `libero` branch is **bs=1 only** — it reverts these):
  1. action-head loss reduction `has_real_action[:, None, None]` in
     `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` (the original `[:, None]`
     crashes at bs>1).
  2. **`enforce_full_chunks`** (new dataset flag). LIBERO samples have a *variable* number of chunks
     near episode/language boundaries (video 17/25/33 frames, state `(2|3|4, 64)`, action `24·n`),
     which is self-consistent at bs=1 but makes the collate `np.stack` fail intermittently when
     batching differently-sized samples. The gated flag (default **off**; **on** for the LIBERO
     config) makes the video/state/action samplers yield only full `max_chunk_size` samples (skips
     boundary samples via the existing `get_step_data() -> None` path), so every sample has identical
     shape and bs>1 batches cleanly — no CPU offload needed. Files:
     `groot/vla/data/dataset/lerobot.py` (param `enforce_full_chunks`),
     `groot/vla/data/dataset/lerobot_sharded.py` (3 samplers),
     `groot/vla/configs/data/dreamzero/libero_relative_wan22.yaml` (`enforce_full_chunks: true`).
- **H200 batch-size sweep (8× H200, ZeRO-2, no offload).** Training is **compute-bound, not
  memory-bound**; per-sample throughput saturates at bs=2 and memory grows slowly:

  | per-device bs | peak mem / GPU | step time | s/sample |
  |---|---|---|---|
  | 1 | 43.1 GB | 1.24 s | 0.155 |
  | 2 | 43.2 GB | 1.59 s | 0.099 |
  | 4 | 44.3 GB | 3.16 s | 0.099 |
  | 8 | 47.4 GB | 6.15 s | 0.096 |
  | 16 | 53.5 GB | 12.39 s | 0.097 |

  Recommendation: `PER_DEVICE_BATCH_SIZE=2` for best throughput; 4–8 also fit comfortably for a
  larger global batch. (Parity vs bs=1 was confirmed via `scripts/test_action_loss_batch.py`
  (exact, rel 0) + matching end-to-end loss magnitudes on the no-offload path.)
- **Eval + upload watcher (branch `libero`).** `eval_utils/watch_and_eval_libero.py` +
  `scripts/eval/watch_eval_libero.sh`: watches a training `OUTPUT_DIR`, runs the §7 LIBERO sim eval
  on each new `checkpoint-N` (policy server on a spare GPU + LIBERO client), logs
  `eval/success_rate` (+ per-task) to a **sibling wandb run `<run>-eval`** (custom `eval/ckpt_step`
  axis; a sibling run because a watcher + training cannot reliably share one *live* run), keeps
  **`latest-N ∪ best-M`** checkpoints in place (originals → full/resumable, deduped), and mirrors
  the best-M to a **HuggingFace Hub** repo (upload on enter best-M, delete on evict, background).
- **Configurable `SAVE_TOTAL_LIMIT`** in `scripts/train/libero_training_wan22.sh` (env
  `SAVE_TOTAL_LIMIT`, default 5). Set it **high** (e.g. `100000`) when using the watcher so the
  watcher is the sole pruner and best checkpoints aren't rotated away.
- **`.gitignore`** now ignores `.env`.

## 10.1 Fresh-instance environment setup

**Prereqs:** CUDA 12.x toolkit at `/usr/local/cuda` (provides `nvcc`); H100/H200 GPUs;
`/root/dreamzero/.env` containing `HF_TOKEN=...` and `WANDB_API_KEY=...`; and for eval rendering on
a busy multi-GPU node, the OSMesa CPU-rendering lib: `apt-get install -y libosmesa6 libgl1-mesa-glx`.

**Install Miniconda (if no conda):**
```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /root/miniconda.sh
bash /root/miniconda.sh -b -p /root/miniconda3
/root/miniconda3/bin/conda init bash && source ~/.bashrc
```

**Env 1 — `dreamzero` (training + serving, GPU, py3.11):**
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n dreamzero python=3.11 -y -c conda-forge --override-channels
conda activate dreamzero
python -m ensurepip --upgrade                       # conda-forge python ships WITHOUT pip
cd /root/dreamzero
export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129   # torch 2.8 cu129
MAX_JOBS=96 python -m pip install --no-build-isolation flash-attn                     # ~10–15 min compile
```

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

## 10.2 Download checkpoints + dataset (and convert)

```bash
conda activate dreamzero && cd /root/dreamzero
set -a; . ./.env; set +a
python -m pip install -q -U huggingface_hub hf_transfer    # provides the `hf` CLI
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p checkpoints data

# (a) Wan2.2-TI2V-5B backbone (~34 GB: DiT shards, Wan2.2_VAE.pth, T5 encoder, umt5 tokenizer)
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
# (b) CLIP image encoder from Wan2.1 (~4.5 GB; Wan2.2-TI2V-5B doesn't ship it)
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
# (c) umt5-xxl tokenizer (bundled inside Wan2.2 — just copy it out)
mkdir -p ./checkpoints/umt5-xxl && cp ./checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl/* ./checkpoints/umt5-xxl/

# (d) LIBERO dataset -> convert to DreamZero MP4 LeRobot format (~11 GB, 1693 eps)
hf download physical-intelligence/libero --repo-type dataset --local-dir ./data/libero_raw_lerobot
python scripts/data/convert_libero_to_dreamzero.py \
    --src data/libero_raw_lerobot --dst data/libero_lerobot --num-workers 64
```

## 10.3 Train

bs=1 (branch `libero`). **Run the env-prefixed launcher as ONE line** — multi-line `\`
continuations often drop the inline `VAR=val ... bash ...` vars when pasted into tmux:

```bash
conda activate dreamzero && cd /root/dreamzero
set -a; . ./.env; set +a                       # HF_TOKEN + WANDB_API_KEY into the env
export WANDB_RUN_ID=dreamzero_libero_wan22      # MUST be unique & never-before-deleted (see gotchas)

NUM_GPUS=7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 SAVE_TOTAL_LIMIT=100000 OUTPUT_DIR=$PWD/checkpoints/dreamzero_libero_wan22 LIBERO_DATA_ROOT=$PWD/data/libero_lerobot PER_DEVICE_BATCH_SIZE=1 MAX_STEPS=100000 SAVE_STEPS=1000 SAVE_STRATEGY=steps TRAIN_ARCH=full PYTHON_BIN=$(which python) bash scripts/train/libero_training_wan22.sh
```

- `NUM_GPUS=7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6` leaves **GPU 7 free for the eval server**. On a
  fully free node (no concurrent eval) use `NUM_GPUS=8` and drop `CUDA_VISIBLE_DEVICES`.
- `SAVE_TOTAL_LIMIT=100000` ⇒ HF never prunes (the watcher prunes). Set `5` (or omit) if not using
  the watcher.
- wandb: needs `WANDB_API_KEY` (from `.env`) + a unique `WANDB_RUN_ID`. To disable wandb entirely:
  prepend `WANDB_MODE=disabled REPORT_TO=none`.
- **Per-device bs>1:** `git checkout libero_distributed`, then the same command with
  `PER_DEVICE_BATCH_SIZE=2` (best throughput; ≤ ~8–16 fit on H200). The `libero` branch is bs=1 only.

## 10.4 Eval + best-checkpoint mirror — automatic watcher (recommended)

Run in a second terminal alongside training. It serves each new checkpoint on `SERVER_GPU`, runs the
sim eval, logs to a **sibling `<run>-eval` wandb run**, keeps `latest-N ∪ best-M` locally
(full/resumable), and mirrors best-M to the Hub. The launcher self-sources conda + `.env`.

```bash
OUTPUT_DIR=/root/dreamzero/checkpoints/dreamzero_libero_wan22 SERVER_GPU=7 TRIALS=3 KEEP_BEST=3 KEEP_LATEST=3 UPLOAD_REPO=<your-hf-user>/dreamzero-libero-best bash /root/dreamzero/scripts/eval/watch_eval_libero.sh --wandb-run-id <training-run-id>
```

Pass `--wandb-run-id` = the training run id; eval is logged to `<training-run-id>-eval` (same
wandb project, so you can overlay `eval/success_rate` with the training curves in one chart).

Knobs (env): `SERVER_GPU` (default 7), `TRIALS` (per task, default 10), `MAX_TASKS` (default **3** =
first 3 tasks of the suite; set `0` for all 10), `KEEP_BEST` (>0 enables in-place retention + best-N),
`KEEP_LATEST` (default 5), `UPLOAD_REPO` (enables HF mirror), `UPLOAD_MODEL_ONLY=1` (~25 GB model-only
instead of full ~130 GB), `SEPARATE_RUN` (default **1** = sibling eval run; `0` = attempt same-run,
not recommended), `MUJOCO_GL_BACKEND` (default `osmesa`; see gotcha #6). Extra flags pass through
(e.g. `--wandb-run-id`, `--task-suite-name`, `--all`, `--exit-when-done`, `--upload-public`).

- **Full checkpoints incl. optimizer are kept/uploaded by default** (resumable). A full ckpt ≈ 130 GB.
- **HF storage:** free private = 100 GB (a single full ckpt won't fit), PRO = 1 TB. For free use
  `UPLOAD_MODEL_ONLY=1` (best-3 ≈ 75 GB).
- **Timing (H200, `libero_spatial`, untrained = worst case, OSMesa CPU render):** server load ~2 min,
  ~2–2.5 min per full 220-step episode ⇒ **~20–25 min for the default 3 trials × 3 tasks**
  (`MAX_TASKS=3`), or ~60–75 min for 3 trials × 10 tasks (`MAX_TASKS=0`). Faster as the policy
  succeeds (episodes end early). `--latest-only` evaluates the newest checkpoint each cycle and skips
  behind to keep up with the ~21 min/1k-step (bs=1) checkpoint cadence.

## 10.5 Eval — manual, single model (two terminals; §7 with /root paths)

Terminal A (server, `dreamzero`):
```bash
conda activate dreamzero && cd /root/dreamzero && set -a; . ./.env; set +a
CUDA_VISIBLE_DEVICES=7 python eval_utils/serve_dreamzero_libero.py \
    --model_path ./checkpoints/dreamzero_libero_wan22 \
    --embodiment_tag libero_sim --tokenizer_path ./checkpoints/umt5-xxl --port 8000
```
Terminal B (client, `dreamzero_libero`):
```bash
conda activate dreamzero_libero && cd /root/dreamzero
MUJOCO_GL=egl python eval_utils/run_libero_eval.py \
    --host 0.0.0.0 --port 8000 --task-suite-name libero_spatial \
    --num-trials-per-task 50 --video-out-path ./eval_outputs/libero_spatial/videos
```
`--model_path` can be the top-level finished model **or** any `checkpoint-N/` dir (both are servable:
they contain `config.json` + consolidated `model*.safetensors` + `experiment_cfg/`).

## 10.6 Download model weights from HF + resume training

**Download** the mirrored best checkpoints from your private repo:
```bash
conda activate dreamzero && cd /root/dreamzero && set -a; . ./.env; set +a
# all best checkpoints:
hf download <your-hf-user>/dreamzero-libero-best --local-dir ./checkpoints/best_from_hf
# or just one: hf download <your-hf-user>/dreamzero-libero-best --include "checkpoint-12000/*" \
#                  --local-dir ./checkpoints/best_from_hf
```

**Resume training** from a downloaded checkpoint (these are FULL checkpoints incl. the DeepSpeed
`global_step*` optimizer state, so training resumes exactly):
```bash
mkdir -p $PWD/checkpoints/resumed
cp -r ./checkpoints/best_from_hf/checkpoint-12000 $PWD/checkpoints/resumed/
# Training auto-resumes from the latest checkpoint-N in OUTPUT_DIR (no resume flag):
NUM_GPUS=7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 SAVE_TOTAL_LIMIT=100000 OUTPUT_DIR=$PWD/checkpoints/resumed LIBERO_DATA_ROOT=$PWD/data/libero_lerobot PER_DEVICE_BATCH_SIZE=1 MAX_STEPS=100000 SAVE_STEPS=1000 SAVE_STRATEGY=steps TRAIN_ARCH=full PYTHON_BIN=$(which python) WANDB_RUN_ID=<unique> bash scripts/train/libero_training_wan22.sh
```
**Resume semantics (automatic, no flag) — see §6.2:** `get_checkpoint_path(OUTPUT_DIR)` →
- top-level `config.json` present ⇒ run is "finished" ⇒ prints "Models is ready … Skip training" and exits;
- else `checkpoint-N/` present ⇒ "Resuming training from …/checkpoint-N" (loads the `global_step*`
  optimizer state) and continues to `MAX_STEPS`;
- else first-time training.

To **serve/eval** a downloaded checkpoint, point `--model_path` at the `checkpoint-N/` dir (§10.5),
or just drop it into the watcher's `OUTPUT_DIR`.

## 10.7 Gotchas added this iteration

1. **conda-forge `python` ships without `pip`** → `python -m ensurepip --upgrade` after `conda create`.
2. **wandb won't reuse a deleted run id** (`HTTP 409: run was previously created and deleted`). Never
   `Api().run(...).delete()` then reuse the same `WANDB_RUN_ID`; pick a fresh id (e.g. `..._v2`).
3. **tmux multi-line paste** with trailing `\` can drop the inline `VAR=val … bash …` env prefixes →
   run the launcher invocation as a **single line**.
4. **Watcher = sole pruner** (with `SAVE_TOTAL_LIMIT` high). It must stay running, else checkpoints
   accumulate (~130 GB each). Normal footprint `latest-N ∪ best-M` ≈ ~1 TB.
5. **HF private storage:** free 100 GB / PRO 1 TB; full ckpt ≈ 130 GB, model-only ≈ 25 GB.
6. **Eval rendering on a busy multi-GPU node.** MuJoCo **EGL** rendering aborts (SIGABRT, exit 134)
   when its GPU is saturated: on the default device 0 (training) it dies at env creation; co-located
   with the policy server on the spare GPU it dies on the **2nd episode** (EGL + the server's CUDA
   context on the same device is unstable across episodes). Fix: render on **CPU via OSMesa** —
   `apt-get install -y libosmesa6` and run the client with `MUJOCO_GL=osmesa`. The watcher
   (`scripts/eval/watch_eval_libero.sh`) **defaults to `MUJOCO_GL_BACKEND=osmesa`** for this reason;
   only set `MUJOCO_GL_BACKEND=egl` if you have a fully-idle GPU dedicated to rendering. (The watcher
   also pins EGL to `SERVER_GPU` when EGL is used.) OSMesa renders on CPU (~2× slower per episode than
   GPU EGL) but is robust and runs on the spare GPU's host, so it doesn't slow training.
7. **wandb eval logging uses a sibling run.** A watcher and training can't both live-write to one
   wandb run — the second writer's points are silently dropped (`train/*` lands, `eval/*` vanishes).
   The watcher logs eval to **`<run>-eval`** by default (`SEPARATE_RUN=1`); overlay it with the
   training run in the wandb UI (same project). Separately, wandb's default x-axis `Step` (`_step`)
   counts `wandb.log()` calls (~2× the real step here, since the trainer logs ~2 things/step) — set
   the chart **X-Axis to `train/global_step`** to match the terminal step.

## 10.8 Reproduction checklist (this iteration)

```text
[ ] Miniconda + `conda init bash`                                                 # §10.1
[ ] env dreamzero (ensurepip, pip install -e . cu129, flash-attn)                 # §10.1
[ ] env dreamzero_libero + ~/.libero/config.yaml                                  # §10.1
[ ] .env has HF_TOKEN + WANDB_API_KEY                                             # §10
[ ] download Wan2.2-TI2V-5B, Wan2.1 CLIP, copy umt5 tokenizer                      # §10.2
[ ] download + convert LIBERO dataset                                             # §10.2
[ ] train (single-line launcher; NUM_GPUS=7, GPU7 free; SAVE_TOTAL_LIMIT high)    # §10.3
[ ] eval watcher (TRIALS/KEEP_BEST/KEEP_LATEST/UPLOAD_REPO) + same --wandb-run-id # §10.4
[ ] (bs>1) git checkout libero_distributed; PER_DEVICE_BATCH_SIZE=2               # §10.0/§10.3
[ ] download best from HF + resume (auto from OUTPUT_DIR)                          # §10.6
```
