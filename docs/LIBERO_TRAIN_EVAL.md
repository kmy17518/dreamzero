# Training and Evaluating DreamZero on LIBERO

This guide explains how to train DreamZero on the **LIBERO** benchmark with the
**Wan2.2-TI2V-5B** backbone (full fine-tune, from scratch, no LoRA) and how to run the
LIBERO simulation evaluation.

It covers the LIBERO-specific pieces added to this repo:

| Piece | File |
|---|---|
| LIBERO → GEAR dataset converter | `scripts/data/convert_libero_to_gear.py` |
| Embodiment tag `libero` | `groot/vla/data/schema/embodiment_tags.py` |
| Projector index + prompt text | `groot/vla/configs/model/dreamzero/transform/base.yaml`, `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py` |
| Modality config + transform | `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` |
| Dataset config (5B) | `groot/vla/configs/data/dreamzero/libero_relative_wan22.yaml` |
| Training script (5B, full, from scratch) | `scripts/train/libero_training_wan22.sh` |
| LIBERO eval server (openpi protocol) | `eval_utils/serve_dreamzero_libero.py` |

---

## 0. Why a converter is needed

The openpi LIBERO LeRobot datasets (e.g. `physical-intelligence/libero`, and the
`libero-10` / `libero-90` conversions under
`/root/cs224r_custom_private/data/libero100`) store camera frames as **inline PNG bytes**
inside the parquet files (`dtype: image`, `total_videos: 0`). DreamZero's data loader
reads frames from **MP4 files** via decord, exactly like the DROID GEAR dataset.

`convert_libero_to_gear.py` bridges this by re-encoding the inline frames as MP4 (one file
per episode × camera, laid out like DROID) and generating the GEAR metadata
(`info.json`, `modality.json`, `embodiment.json`, `stats.json`, `tasks.jsonl`,
`episodes.jsonl`). It is CFR at the dataset FPS so decord frame timestamps line up with the
parquet `timestamp` column.

LIBERO layout:
- **Cameras (2):** `image` (agentview/exterior) → `video.image`; `wrist_image` (eye-in-hand) → `video.wrist_image`
- **State (8):** eef_pos(3) + eef_axisangle(3) + gripper_qpos(2)
- **Action (7):** delta end-effector pose(6) + gripper(1) — a *command*, not an absolute
  joint target, so we use **`relative_action: false`** (actions are normalized directly).

---

## 1. Convert the dataset

```bash
# libero-10 (already done as a sanity check):
python scripts/data/convert_libero_to_gear.py \
    --src /root/cs224r_custom_private/data/libero100/libero-10 \
    --out /root/dreamzero/data/libero10_gear \
    --embodiment-tag libero --num-workers 8

# libero-90 (your training set; same structure as libero-10):
python scripts/data/convert_libero_to_gear.py \
    --src /root/cs224r_custom_private/data/libero100/libero-90 \
    --out /root/dreamzero/data/libero90_gear \
    --embodiment-tag libero --num-workers 8
```

Output layout (DROID-identical):

```
libero90_gear/
├── data/chunk-000/episode_000000.parquet      # state, actions, timestamp, ... (no image cols)
├── videos/chunk-000/observation.images.image/episode_000000.mp4
├── videos/chunk-000/observation.images.wrist_image/episode_000000.mp4
└── meta/{info.json, modality.json, embodiment.json, stats.json, tasks.jsonl, episodes.jsonl}
```

The converter uses `imageio`/ffmpeg to write `libx264 yuv420p` MP4 — verify decord can
read them and that frame N matches the original PNG (the `libero10_gear` output was checked:
frame count matches, timestamps align, mean abs pixel diff ≈ 1.3/255 from H.264).

---

## 2. What was registered for the `libero` embodiment

1. **Embodiment tag** `LIBERO = "libero"` in `groot/vla/data/schema/embodiment_tags.py`.
2. **Projector index** `libero: 33` in
   `groot/vla/configs/model/dreamzero/transform/base.yaml`. (The model forces
   `embodiment_id = 0` internally for the state/action encoders, so this index only selects
   the language-prompt template in `collate()`.)
3. **Prompt template** for `libero` added to `collate()` in
   `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py` (otherwise unknown
   embodiments raise `ValueError`). The 2 views are tiled into the standard 2×2 grid:
   top-left = agentview, bottom-left = wrist, right column = black.
4. **Modality config + transform** (`modality_config_libero`, `transform_libero`) and the
   `modality_configs` / `transforms` / `metadata_versions` / `fps` maps in
   `base_48_wan_fine_aug_relative.yaml`.
5. **Dataset config** `libero_relative_wan22.yaml` (160×320 video, `relative_action: false`).

Validated: the full data pipeline composes and yields a model-ready batch
(`images: (33, 320, 640, 3)`, `embodiment_id: 33`, task text resolved from `task_index`).

---

## 3. Train (Wan2.2-5B, full fine-tune, from scratch, no LoRA)

```bash
# Smoke test (1 GPU, few steps):
DATA_ROOT=/root/dreamzero/data/libero10_gear MAX_STEPS=10 NUM_GPUS=1 \
    bash scripts/train/libero_training_wan22.sh

# Full training on libero-90 (8 GPUs):
DATA_ROOT=/root/dreamzero/data/libero90_gear NUM_GPUS=8 MAX_STEPS=100000 \
    bash scripts/train/libero_training_wan22.sh
```

Key settings (mirrors `droid_training_full_finetune_wan22.sh` but for LIBERO):
- `train_architecture=full`, `save_lora_only=false` → **full fine-tune, no LoRA**
- **No `pretrained_model_path`** → action/state heads trained **from scratch**; only the
  Wan2.2-TI2V-5B base video weights are loaded as the DiT/VAE init.
- `model/dreamzero/action_head=wan_flow_matching_action_tf_wan22` → 5B backbone (dim 3072,
  30 layers, 48-channel VAE38, 160×320, `frame_seqlen=50`).
- `num_views=2`, `num_frames=33`, `action_horizon=24`, `num_frame_per_block=2`.

Weights (downloaded automatically if missing, or pre-place under `./checkpoints`):
- `Wan-AI/Wan2.2-TI2V-5B` (diffusion + T5 + VAE)
- `Wan-AI/Wan2.1-I2V-14B-480P` (only the CLIP file `models_clip_open-clip-...-vit-huge-14.pth`)
- `google/umt5-xxl` (tokenizer)

---

## 4. Evaluate in LIBERO simulation

Evaluation uses openpi's LIBERO client
(`third_party/openpi/examples/libero/main.py`) talking to a DreamZero policy server over the
**openpi websocket protocol**. We provide that server in
`eval_utils/serve_dreamzero_libero.py`.

### 4a. Start the DreamZero policy server (GPU box, this repo's env)

```bash
# Serve a LIBERO-trained DreamZero 5B checkpoint:
python eval_utils/serve_dreamzero_libero.py \
    --model_path ./checkpoints/dreamzero_libero_wan22_full_finetune \
    --embodiment_tag libero --port 8000
```

The server speaks the protocol openpi's client expects: it receives
`observation/image`, `observation/wrist_image`, `observation/state`, `prompt` and returns
`{"actions": (N, 7)}`. By default it runs **stateless** (resets the causal KV cache each
call), which matches openpi's replanning client that sends no reset signal.

### 4b. Run the LIBERO client (openpi LIBERO env)

The LIBERO simulator (robosuite + mujoco + BDDL) is a separate environment — follow
`third_party/openpi/examples/libero/README.md`. Easiest is Docker:

```bash
sudo xhost +local:docker
# point the client at the DreamZero server started above
CLIENT_ARGS="--args.task-suite-name libero_10 --args.host <server_host> --args.port 8000" \
  docker compose -f third_party/openpi/examples/libero/compose.yml up --build
```

Or without Docker (separate venv, python 3.8):

```bash
cd third_party/openpi
uv venv --python 3.8 examples/libero/.venv && source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python examples/libero/main.py --args.task-suite-name libero_10 --args.port 8000
# (MUJOCO_GL=glx ... if you hit EGL errors)
```

Rollout videos + success rate are written by the client (`data/libero/videos`).

---

## 5. Verify the harness with the released 14B DROID checkpoint

To confirm the **eval plumbing runs end-to-end** before you have a LIBERO-trained
checkpoint, serve the released 14B `DreamZero-DROID` checkpoint and point the LIBERO client
at it. `--embodiment_tag oxe_droid` makes the server map LIBERO's 2 cameras onto DROID's 3
views (agentview duplicated) and truncate the model's 8-dim joint+gripper output to LIBERO's
7-dim action space. The actions are not meaningful for LIBERO (wrong embodiment) — this only
verifies load → observe → infer → act → step.

```bash
hf download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir ./checkpoints/DreamZero-DROID

python eval_utils/serve_dreamzero_libero.py \
    --model_path ./checkpoints/DreamZero-DROID --embodiment_tag oxe_droid --port 8000
# then run the LIBERO client from section 4b against port 8000
```

---

## How the `libero_eval` simulator env was created

The LIBERO mujoco simulator runs in its own conda env (Python 3.8), separate from the
training/serving env. Exact steps used on this machine:

```bash
# 1. Create the env
conda create -y -n libero_eval python=3.8
PY=/root/miniforge3/envs/libero_eval/bin/python
ENVBIN=/root/miniforge3/envs/libero_eval/bin

# 2. cmake is required to build egl_probe (a robosuite dep). Use cmake 3.x:
#    (cmake 4.x errors on egl_probe's old `cmake_minimum_required`)
$PY -m pip install "cmake==3.28.4"          # provides $ENVBIN/cmake on PATH

# 3. Install LIBERO + client requirements (egl_probe build needs $ENVBIN on PATH)
cd /root/cs224r_custom_private/third_party/openpi
PATH=$ENVBIN:$PATH $PY -m pip install \
    -r examples/libero/requirements.txt \
    -r third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 4. Install the LIBERO package + openpi client (no-deps to keep pinned versions)
$PY -m pip install --no-deps -e third_party/libero -e packages/openpi-client
$PY -m pip install msgpack msgpack-numpy websockets

# 5. LIBERO prompts interactively for a dataset path on first import. Pre-create its
#    config so imports are non-interactive (datasets path can be absent -- eval doesn't
#    need LIBERO's demo datasets, only bddl_files / init_files / assets which ship with it):
ROOT=/root/cs224r_custom_private/third_party/openpi/third_party/libero/libero/libero
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<EOF
benchmark_root: $ROOT
bddl_files: $ROOT/bddl_files
init_states: $ROOT/init_files
datasets: $ROOT/../datasets
assets: $ROOT/assets
EOF
```

Headless rendering: this host has NVIDIA + Mesa EGL (`libEGL_nvidia.so`, glvnd vendor
configs), so run the client with `MUJOCO_GL=egl` (use `MUJOCO_GL=glx` if you hit EGL
errors). Verified: `OffScreenRenderEnv` renders `(256,256,3)` agentview + wrist frames and
`get_task_init_states` returns `(50, 123)`. A harmless `EGLError` may print on process exit
(context teardown).

## Environment notes

- `flash_attn` is **not required**: every attention path falls back to
  `torch.nn.functional.scaled_dot_product_attention` when flash-attn is absent (H200/SDPA).
- The conversion + data pipeline run in the `piwan` conda env. Packages added during setup:
  `hydra-core omegaconf dm_tree opencv-python-headless matplotlib av h5py loguru termcolor
  albumentations ftfy regex tianshou==0.5.1 gymnasium tyro msgpack-numpy` and the vendored
  `openpi-client` (`pip install -e third_party/openpi/packages/openpi-client`).
- The LIBERO simulator itself is intentionally kept in openpi's separate
  python-3.8/cu113 environment (see section 4b) — do not install it into the training env.
