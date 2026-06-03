# LIBERO on DreamZero — Fresh-GPU Handoff Runbook

> Purpose: this box is an **on-demand GPU that is being shut down**. This file tells the
> next agent/person exactly what to read and do to set up and run **LIBERO training + eval**
> on a brand-new GPU. The deep details live in [`docs/LIBERO_TRAIN_EVAL.md`](docs/LIBERO_TRAIN_EVAL.md);
> this file is the high-level checklist plus the "what won't survive the shutdown" warnings.

---

## TL;DR — "Do I have everything needed to run on another GPU?"

**Code: yes.** Everything code-related is committed and pushed:
- `dreamzero` repo → `origin/backup` (`git@github.com:kmy17518/dreamzero.git`). The LIBERO
  converter, embodiment tag, configs, training script, eval server, and docs are all in.
- The sibling helper repo `cs224r_custom_private` → `origin/prime-intellect`
  (`git@github.com:kmy17518/cs224r_custom_private.git`, clean & pushed). It provides the
  `openpi` / `libero` simulator submodules and the dataset download/convert scripts.

**Data + environments: NO — these are local to this box and will be lost.** You must either
persist them before shutdown (below) or recreate them on the new GPU:
- Converted GEAR datasets (only `libero10_gear` exists; **`libero90_gear` was never built**).
- Source LIBERO datasets (~75 GB) under `cs224r_custom_private/data/libero100/`.
- The three conda envs (`piwan` train/serve, `libero_eval` sim, `libero_convert`).
- `~/.libero/config.yaml` (sim config).

---

## Read these, in order

1. **This file** — the checklist + gotchas.
2. **[`docs/LIBERO_TRAIN_EVAL.md`](docs/LIBERO_TRAIN_EVAL.md)** — the authoritative end-to-end
   guide: dataset conversion, what was registered for the `libero` embodiment, training,
   eval server, and exact commands used to build the `libero_eval` simulator env.
3. **[`README.md`](README.md)** — base DreamZero install (conda env, `pip install -e .`, flash-attn).
4. Background only if needed: [`docs/WAN22_BACKBONE.md`](docs/WAN22_BACKBONE.md),
   [`docs/DATASET_TO_GEAR_AND_TRAIN.md`](docs/DATASET_TO_GEAR_AND_TRAIN.md),
   [`docs/DROID_CONVERSION.md`](docs/DROID_CONVERSION.md).

---

## What persists vs. what is lost on shutdown

| Item | Location on this box | Persists? | How to recover on new GPU |
|---|---|---|---|
| dreamzero code | `/root/dreamzero` | ✅ pushed `origin/backup` | `git clone -b backup ...` |
| cs224r helper repo (openpi/libero, scripts) | `/root/cs224r_custom_private` | ✅ pushed `origin/prime-intellect` | `git clone --recurse-submodules -b prime-intellect ...` |
| Source LIBERO datasets (~75 GB: libero-10 12G, libero-90 63G) | `/root/cs224r_custom_private/data/libero100/` | ❌ local only | re-download (RLDS from HF) + re-convert |
| GEAR dataset `libero10_gear` (1.3 GB) | `/root/dreamzero/data/libero10_gear` | ❌ local only | re-run converter |
| GEAR dataset `libero90_gear` (training set) | — | ❌ **never built** | must build (see step 3) |
| conda env `piwan` (train + serve) | `/root/miniforge3/envs/piwan` | ❌ local | recreate per README |
| conda env `libero_eval` (sim, py3.8) | `/root/miniforge3/envs/libero_eval` | ❌ local | recreate per LIBERO_TRAIN_EVAL.md §"How the libero_eval env was created" |
| `~/.libero/config.yaml` | `~/.libero/` | ❌ local | recreate per LIBERO_TRAIN_EVAL.md §"How the libero_eval env was created" |
| Model weights (Wan2.2-TI2V-5B, CLIP, umt5) | `./checkpoints/` | ❌ local | auto-downloaded by training/serve scripts |

---

## OPTIONAL — before you shut down THIS box (saves hours of re-download/convert)

Converting libero-90 from raw RLDS is large (~75 GB download) and slow. If you can attach a
**persistent/network disk** or push to **HuggingFace/cloud storage**, snapshot the converted
GEAR datasets now so the next GPU skips conversion entirely:

```bash
# Converted (model-ready) datasets — most valuable to keep:
#   /root/dreamzero/data/libero10_gear        (1.3 GB, exists)
#   /root/dreamzero/data/libero90_gear        (build first, see step 3, then save it too)
# Source LeRobot datasets (if you want to avoid re-downloading raw RLDS):
#   /root/cs224r_custom_private/data/libero100/{libero-10,libero-90}

# Example: push converted datasets to HF (private dataset repo)
huggingface-cli upload <you>/dreamzero-libero-gear /root/dreamzero/data/libero10_gear libero10_gear --repo-type dataset
# (repeat for libero90_gear once built)
```

If you skip this, everything is still recoverable from public HF sources — it just costs the
download + conversion time on the new box.

---

## Fresh-GPU setup (new machine)

Assumes the new box has NVIDIA GPUs + conda. Clone the two repos **as siblings** under the
same parent dir (the eval server resolves openpi at
`<parent>/cs224r_custom_private/third_party/openpi/src`):

```bash
cd /root   # or any parent dir, but keep the two repos side-by-side
git clone -b backup git@github.com:kmy17518/dreamzero.git
git clone --recurse-submodules -b prime-intellect git@github.com:kmy17518/cs224r_custom_private.git
```

### 1. Training/serving env  (see `README.md` for the canonical steps)

```bash
conda create -n dreamzero python=3.11 && conda activate dreamzero
cd /root/dreamzero
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
# flash-attn is OPTIONAL — code falls back to torch SDPA if absent (see LIBERO_TRAIN_EVAL.md §Environment notes)
pip install "huggingface_hub[cli]"
# extra deps used by the LIBERO data pipeline / eval server:
pip install hydra-core omegaconf dm_tree opencv-python-headless matplotlib av h5py loguru \
    termcolor albumentations ftfy regex tianshou==0.5.1 gymnasium tyro msgpack-numpy
pip install -e /root/cs224r_custom_private/third_party/openpi/packages/openpi-client
```

### 2. Get the data

If you persisted the converted GEAR datasets, just copy them back to
`/root/dreamzero/data/` and skip to step 3.

Otherwise, recreate them (download raw → LeRobot → GEAR). Set `HF_TOKEN` first
(`cs224r_custom_private/.env` had one on the old box; you'll need your own):

**Where libero-100 comes from** (downloaded by `scripts/download_libero100_local.py`; all
"no_noops" = the OpenVLA regeneration: 256px, 180° rotation, no-op + failed-trajectory
filtering):

| Suite | HF source repo | Notes |
|---|---|---|
| `libero_10_no_noops` | `openvla/modified_libero_rlds` (dataset) | canonical; the exact source `physical-intelligence/libero` was converted from |
| `libero_90_no_noops` | `real-lab/libero_filtered_noops_rlds_dataset` (dataset) | OpenVLA's RLDS repo has no libero_90; this regenerates it. Its `wrist_image` is 128×128 (vs libero_10's 256×256) — the converter upscales the wrist to 256×256 for a uniform schema |
| reference (verification only) | `physical-intelligence/libero` (dataset) | LeRobot dataset used to check the conversion |

```bash
cd /root/cs224r_custom_private
# a) download raw RLDS + reference LeRobot (~75 GB) from the HF repos in the table above
python scripts/download_libero100_local.py
# b) RLDS -> LeRobot (produces data/libero100/{libero-10,libero-90})
python scripts/convert_libero100_to_lerobot.py        # see script header for args
# c) LeRobot -> GEAR (dreamzero format). THIS is what training consumes:
cd /root/dreamzero
python scripts/data/convert_libero_to_gear.py \
    --src /root/cs224r_custom_private/data/libero100/libero-10 \
    --out /root/dreamzero/data/libero10_gear --embodiment-tag libero --num-workers 8
python scripts/data/convert_libero_to_gear.py \
    --src /root/cs224r_custom_private/data/libero100/libero-90 \
    --out /root/dreamzero/data/libero90_gear --embodiment-tag libero --num-workers 8
```

> ⚠️ `libero90_gear` (the actual training set) was **never built on the old box** — only
> `libero10_gear` (used as a sanity check) existed. You must build it before full training.

### 3. Train (Wan2.2-5B, full fine-tune, from scratch — weights auto-download)

```bash
# Smoke test (1 GPU, 10 steps) against the small set:
DATA_ROOT=/root/dreamzero/data/libero10_gear MAX_STEPS=10 NUM_GPUS=1 \
    bash scripts/train/libero_training_wan22.sh

# Full training on libero-90 (8 GPUs):
DATA_ROOT=/root/dreamzero/data/libero90_gear NUM_GPUS=8 MAX_STEPS=100000 \
    bash scripts/train/libero_training_wan22.sh
```

Checkpoints land in `./checkpoints/dreamzero_libero_wan22_full_finetune`. See
`docs/LIBERO_TRAIN_EVAL.md` §3 for what each flag means.

### 4. Evaluate in the LIBERO simulator

Two pieces: (a) the DreamZero policy **server** (in the `dreamzero`/`piwan` env), and
(b) the openpi LIBERO **client** (in a separate py3.8 sim env). Full instructions in
`docs/LIBERO_TRAIN_EVAL.md` §4 and the §"How the libero_eval env was created" appendix
(includes the `~/.libero/config.yaml` you must recreate and `MUJOCO_GL=egl|glx`).

```bash
# (a) serve a LIBERO-trained checkpoint:
python eval_utils/serve_dreamzero_libero.py \
    --model_path ./checkpoints/dreamzero_libero_wan22_full_finetune \
    --embodiment_tag libero --port 8000

# (b) run the client from the openpi libero env (see LIBERO_TRAIN_EVAL.md §4b)
```

To validate the eval plumbing **before** you have a LIBERO checkpoint, serve the released 14B
DROID checkpoint with `--embodiment_tag oxe_droid` (see `docs/LIBERO_TRAIN_EVAL.md` §5).

---

## Critical gotchas (don't get burned)

1. **`libero90_gear` does not exist yet** — only `libero10_gear` was built. Build the 90-task
   set before real training (step 2c).
2. **openpi/libero live in the sibling repo**, not in dreamzero. Clone
   `cs224r_custom_private` **next to** dreamzero so `eval_utils/serve_dreamzero_libero.py` can
   find `../cs224r_custom_private/third_party/openpi/src` (or pip-install openpi-client).
3. **Two separate environments**: training/serving (py3.11) vs. the LIBERO simulator
   (py3.8/cu113). Do **not** install the simulator into the training env.
4. **Recreate `~/.libero/config.yaml`** or LIBERO imports block interactively (see the eval
   appendix in `docs/LIBERO_TRAIN_EVAL.md`).
5. **flash-attn optional**: everything falls back to torch SDPA; don't block on building it.
6. **HF_TOKEN required** for the raw RLDS downloads in step 2.
