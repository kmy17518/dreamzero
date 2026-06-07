# DreamZero on LIBERO with **Explicit Conditioning** (future-frame shift, Wan2.2-TI2V-5B)

This guide documents the **explicit-conditioning** variant of DreamZero‑on‑LIBERO: a one‑block
**future‑frame shift** that makes the joint video+action DiT predict the video **one step further
into the future than the action**, and feed its own previously‑generated frame back in as context.

It is a thin, fully **gated** addition on top of the baseline LIBERO/Wan2.2‑5B pipeline documented in
[`docs/LIBERO.md`](./LIBERO.md). The baseline training/eval paths are **untouched**; this experiment
runs from its own launcher + action‑head config. Read `LIBERO.md` first for the shared environment,
dataset conversion, and watcher details — this file only covers the **delta**.

> Branch: `libero_explicit_conditioning`. Flag: `future_frame_shift` (on the action‑head config).

---

## 0. TL;DR — what was built

| Area | File |
|---|---|
| Flag + train‑time register shift | `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` (`WANPolicyHeadConfig.future_frame_shift` / `future_frame_shift_blocks`; shift in `forward()`) |
| Action‑head config (enables the flag) | `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf_wan22_future_shift.yaml` |
| Training launcher | `scripts/train/libero_training_wan22_future_shift.sh` |
| Eval priming (server) | `eval_utils/serve_dreamzero_libero.py` (auto‑detects the flag; primes the rollout) |

The data pipeline, resolution (160×320), VAE, block sizes (`num_frame_per_block=2`,
`num_action_per_block=24`, `num_state_per_block=1`, `num_frames=33`, `max_chunk_size=4`), and the
DiT attention are **unchanged**. The behavioral change is entirely a **re‑indexing of the action /
state register** (training) plus a **priming wrapper** (eval).

---

## 1. Concept: what differs from the original

### 1.1 Original DreamZero (baseline)

DreamZero is **autoregressive over blocks**. A block is one closed‑loop unit = `num_frame_per_block`
latent video frames ≈ one `num_action_per_block` action chunk ≈ one state token. For LIBERO/5B a
33‑frame clip → 9 latent frames → a leading context frame + **4 blocks**.

Per register slot `b` the joint DiT denoises a **video block and its action chunk together**,
attending causally to the **clean** earlier video blocks (teacher forcing). Video, action, and state
at slot `b` are the **same window `b`** — "aligned next video frames & action chunks":

| slot `b` | VIDEO target | ACTION target | STATE token | attends to clean video |
|---|---|---|---|---|
| 0 | `o_0` | `a_0` | `s_0` | `o_ctx` |
| 1 | `o_1` | `a_1` | `s_1` | `o_ctx, o_0` |
| 2 | `o_2` | `a_2` | `s_2` | `o_ctx, o_0, o_1` |
| 3 | `o_3` | `a_3` | `s_3` | `o_ctx, o_0, o_1, o_2` |

(`o_b` = the 2 latent frames of window `b`; `o_ctx` = leading context frame; `a_b` = the 24‑action
chunk of window `b`; `s_b` = proprio at the start of window `b`.)

### 1.2 Explicit conditioning (this variant)

We want, per closed‑loop step `t`:

- **input**: the current observation `o_t` **temporally concatenated with the previously‑generated
  next frame `ô_{t+1}`** (produced one step earlier);
- **outputs**: the **action for `t+1`** and the **video for `t+2`**.

So the **video leads the action/state pair by exactly one block**. Realized by **rolling the
action+state register back one block** relative to the (unchanged) video target — slot `b` becomes
`(o_b, a_{b-1}, s_{b-1})`:

| slot `b` | VIDEO target | ACTION target | STATE token | attends to clean video (= current obs **++** generated next frame) |
|---|---|---|---|---|
| 0 | `o_0` | — (masked) | — | `o_ctx` |
| 1 | `o_1` | `a_0` | `s_0` | `o_ctx, o_0` |
| 2 | `o_2` | `a_1` | `s_1` | `o_ctx, o_0, o_1` |
| 3 | `o_3` | `a_2` | `s_2` | `o_ctx, o_0, o_1, o_2` |

Reading slot `b` with `t = b−2`: action `a_{b-1}` = `a_{t+1}`, video `o_b` = `o_{t+2}`, state
`s_{b-1}` = `s_{t+1}`, and the clean context `o_0…o_{b-1}` contains the current obs **and** the
just‑generated `o_{t+1}` (`= o_{b-1}`). The action chunk and its state stay co‑located (window
`b−1`), exactly mirroring the baseline's `(o_b, a_b, s_b)` — only shifted back one block.

**Why state = `s_{b-1}` (with the action), not the current `s_{b-2}`:** at eval the model imagines
future *frames* but not future *proprio*. The action `a_{b-1}` is decided when the most recent
**real** observation is window `b−1`, whose real proprio `s_{b-1}` is available. So pairing the action
with `s_{b-1}` is train/eval consistent — **no future proprio is ever fed** (the baseline likewise
feeds the current real proprio every step).

**Cost:** the leading block (`a_{-1}`) has no valid target and is masked out of the action loss, so
one action chunk per clip is unsupervised (3 of 4 with `max_chunk_size=4`). Bump `max_chunk_size`
(and `num_frames = 8·K + 1`) in the launcher to recover it.

### 1.3 Training — inputs / outputs

Teacher‑forced (the fed‑back `ô_{t+1}` is the ground‑truth `o_{t+1}`):

- **Inputs** to the joint DiT: clean GT video latents (all blocks) + **noisy** action/state register
  (rolled back one block) + text/CLIP conditioning + sampled diffusion timesteps.
- **Outputs / loss**: `dynamics_loss` (video flow‑matching, **unchanged** — same target, same
  pipeline) + `action_loss` (flow‑matching on the **shifted** action chunk `a_{b-1}`, with the
  leading slot masked). The video diffusion is identical to baseline; only the action/state tensors
  and the action mask are rolled (`torch.roll` by one block, leading block zeroed/masked) — tensor
  shapes are preserved so all downstream asserts hold.

### 1.4 Eval — inputs / outputs (anticipatory rollout)

The serve wrapper **auto‑detects** `future_frame_shift` from the checkpoint config and primes the
rollout. Per query the client sends the current real frames + 8‑dim proprio + prompt + `session_id`;
the action head conditions on the (re‑encoded real) obs + KV cache and emits one video block + one
action chunk. Because the model now emits the action **one block behind the video**, the first
generated block of every autoregressive sequence carries the unsupervised `a_{-1}`. A *fresh
sequence* occurs at **episode start** and at **every KV‑cache reset** (`current_start_frame >=
local_attn_size`, i.e. roughly every `max_chunk_size` queries). Detected via
`current_start_frame == 1 + num_frame_per_block`:

- **episode start** (single obs frame): return a **no‑op** chunk for that one query (the cache
  advances; the next query returns the first aligned chunk `a_0`).
- **mid‑episode reset** (≥ a full obs chunk available): **regenerate once** on the same obs
  (one extra forward) to fetch the aligned `a_0` — avoids a recurring per‑reset idle.

Net effect: each query returns the **correctly‑aligned** action with **no control lag** (only a
single ~`replan_steps` no‑op at the very start of an episode). Output is `{"actions": (N, 7)}`.

To force the behavior regardless of the checkpoint config, pass `--future_frame_shift True` to the
server.

---

## 2. Environment setup (fresh, empty instance)

Identical to the baseline — follow **`docs/LIBERO.md` §10.1** in full. Summary:

```bash
# Miniconda (if absent)
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /root/miniconda.sh
bash /root/miniconda.sh -b -p /root/miniconda3
/root/miniconda3/bin/conda init bash && source ~/.bashrc

# Env 1 — dreamzero (training + policy serving, GPU, py3.11)
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n dreamzero python=3.11 -y -c conda-forge --override-channels
conda activate dreamzero && python -m ensurepip --upgrade
cd /root/libero_explicit_conditioning
export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129   # torch 2.8 cu129
MAX_JOBS=96 python -m pip install --no-build-isolation flash-attn

# Env 2 — dreamzero_libero (LIBERO MuJoCo sim client, CPU, py3.10)  -- see LIBERO.md §10.1 for the
# exact pinned versions (robosuite 1.4.1 / mujoco 3.2.3 / gym 0.25.2 / LIBERO editable), plus
# ~/.libero/config.yaml. For eval rendering on a busy multi-GPU node also: apt-get install -y libosmesa6
```

`.env` at the repo root must contain `HF_TOKEN=...` and `WANDB_API_KEY=...` (not auto‑loaded; every
shell does `set -a; . ./.env; set +a`).

### 2.1 Weights + dataset

Same as **`docs/LIBERO.md` §10.2** (Wan2.2‑TI2V‑5B backbone, Wan2.1 CLIP, umt5 tokenizer copy, and
the converted MP4 LeRobot dataset). The explicit‑conditioning experiment uses the **same** weights
and the **same** `data/libero_lerobot` — no new downloads or re‑conversion.

```bash
conda activate dreamzero && cd /root/libero_explicit_conditioning && set -a; . ./.env; set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
hf download Wan-AI/Wan2.1-I2V-14B-480P --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
mkdir -p ./checkpoints/umt5-xxl && cp ./checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl/* ./checkpoints/umt5-xxl/
hf download physical-intelligence/libero --repo-type dataset --local-dir ./data/libero_raw_lerobot
python scripts/data/convert_libero_to_dreamzero.py \
    --src data/libero_raw_lerobot --dst data/libero_lerobot --num-workers 64
```

---

## 3. Train (explicit conditioning)

Run the **new** launcher (full fine‑tune, bs=1). It selects
`model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_future_shift` (which sets
`future_frame_shift: true`) and defaults to a **separate** `OUTPUT_DIR` / wandb project so it never
clobbers the baseline run. Run it as **one line** (tmux can drop inline `VAR=val … bash …` prefixes
across line continuations):

```bash
conda activate dreamzero && cd /root/libero_explicit_conditioning
set -a; . ./.env; set +a
export WANDB_RUN_ID=dreamzero_libero_wan22_future_shift   # unique & never-before-deleted

NUM_GPUS=8 SAVE_TOTAL_LIMIT=100000 OUTPUT_DIR=$PWD/checkpoints/dreamzero_libero_wan22_future_shift LIBERO_DATA_ROOT=$PWD/data/libero_lerobot PER_DEVICE_BATCH_SIZE=1 MAX_STEPS=100000 SAVE_STEPS=1000 SAVE_STRATEGY=steps TRAIN_ARCH=full PYTHON_BIN=$(which python) bash scripts/train/libero_training_wan22_future_shift.sh
```

- To leave a GPU free for a co‑located eval watcher, use `NUM_GPUS=7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6`
  (the watcher serves on GPU 7).
- **Full fine‑tune needs ZeRO‑2 sharding across multiple GPUs.** A single‑GPU full FT OOMs at the
  optimizer step (all fp32 master + Adam moments for 5B land on one GPU). Use ≥2 GPUs, or
  `gradient_accumulation_steps` / more GPUs to grow the global batch.
- Disable wandb with `WANDB_MODE=disabled REPORT_TO=none`.
- Resume is automatic from `OUTPUT_DIR` (a top‑level `config.json` means "finished"; otherwise it
  resumes the latest `checkpoint-N`). Same semantics as `LIBERO.md` §6.2 / §10.6.
- Recover the masked leading chunk by bumping `max_chunk_size` (and set `num_frames = 8·max_chunk_size + 1`)
  in the launcher.

Each saved `checkpoint-N` records `future_frame_shift: true` in its config, so **eval auto‑primes**.

`loss_log.jsonl` records `dynamics_loss` (video, unchanged) and `action_loss` (shifted).

---

## 4. Eval

The server (`eval_utils/serve_dreamzero_libero.py`) **auto‑detects** `future_frame_shift` from the
served checkpoint's config and applies the priming/regeneration described in §1.4 — so **both** the
automatic watcher and standalone eval work with **no extra flags**.

### 4.1 Automatic watcher (recommended)

Same as `LIBERO.md` §10.4 — point it at the explicit‑conditioning `OUTPUT_DIR`. The watcher starts
the policy server (which auto‑primes), runs the sim eval per new `checkpoint-N`, logs
`eval/success_rate` to a sibling `<run>-eval` wandb run, retains `latest‑N ∪ best‑M`, and optionally
mirrors best‑M to the Hub. No change is required for explicit conditioning.

```bash
conda activate dreamzero && cd /root/libero_explicit_conditioning
OUTPUT_DIR=$PWD/checkpoints/dreamzero_libero_wan22_future_shift \
SERVER_GPU=7 TRIALS=10 MAX_TASKS=3 KEEP_BEST=3 KEEP_LATEST=3 \
UPLOAD_REPO=<your-hf-user>/dreamzero-libero-future-shift-best \
bash scripts/eval/watch_eval_libero.sh --wandb-run-id dreamzero_libero_wan22_future_shift
```

Knobs (env): `SERVER_GPU` (default 7), `TRIALS`, `MAX_TASKS` (default 3; `0` = all 10 tasks),
`KEEP_BEST`/`KEEP_LATEST`, `UPLOAD_REPO`, `UPLOAD_MODEL_ONLY=1`, `MUJOCO_GL_BACKEND` (default
`osmesa`). Extra flags pass through (`--wandb-run-id`, `--task-suite-name`, `--all`,
`--exit-when-done`, …). See `LIBERO.md` §10.4 / §10.7 for the watcher gotchas (sibling eval run,
OSMesa rendering on busy nodes, watcher‑as‑sole‑pruner).

### 4.2 Standalone / manual eval (two terminals)

**Terminal A — policy server** (`dreamzero`, GPU):

```bash
conda activate dreamzero && cd /root/libero_explicit_conditioning && set -a; . ./.env; set +a
CUDA_VISIBLE_DEVICES=7 python eval_utils/serve_dreamzero_libero.py \
    --model_path ./checkpoints/dreamzero_libero_wan22_future_shift \
    --embodiment_tag libero_sim --tokenizer_path ./checkpoints/umt5-xxl --port 8000
    # future_frame_shift is auto-detected from the checkpoint; add `--future_frame_shift True` to force.
```

On startup it logs `future_frame_shift ON: priming first query of each episode …`. During a rollout
it logs `[future_frame_shift] episode start: returning no-op priming chunk` once per episode and
`[future_frame_shift] cache reset (cf=…): regenerating aligned chunk a_0` at each KV‑cache reset.

**Terminal B — sim client** (`dreamzero_libero`, CPU):

```bash
conda activate dreamzero_libero && cd /root/libero_explicit_conditioning
MUJOCO_GL=osmesa python eval_utils/run_libero_eval.py \
    --host 0.0.0.0 --port 8000 --task-suite-name libero_spatial \
    --num-trials-per-task 50 --video-out-path ./eval_outputs/libero_spatial_future_shift/videos
```

`--model_path` can be the finished top‑level model **or** any `checkpoint-N/` dir. Task suites:
`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`. Use `MUJOCO_GL=glx` only
on a fully‑idle GPU; `osmesa` (CPU) is robust on busy multi‑GPU nodes (see `LIBERO.md` §10.7 #6).

---

## 5. Verification done

- **Training shift** (GPU 7 smoke): the new `forward()` shift ran through forward + loss + backward
  with no shape/roll/assert errors. (A *single‑GPU* full FT then OOMs at the DeepSpeed optimizer step
  — expected; ZeRO‑2 needs ≥2 GPUs. The shift code itself is validated.)
- **Hydra compose**: the new config resolves `future_frame_shift=True` / `future_frame_shift_blocks=1`
  with the Wan2.2‑5B params intact (dim 3072, in_dim 48, 160×320); the baseline config resolves the
  flag to `False`.
- **Eval priming** (GPU 7, served a real 5B checkpoint with the flag, drove the LIBERO sim ~60 steps):
  `future_frame_shift ON` detected; **1** episode‑start idle + **3** mid‑episode regenerate, matching
  **4** total KV‑cache resets (every fresh sequence handled, no `a_{-1}` leaks); 15 generations =
  12 queries + 3 regen double‑calls; **no errors**; video + metrics written.

> The eval test above forced the flag on a baseline (non‑shift) checkpoint — it validates the
> **control flow** (priming/regeneration). A real success‑rate eval is only meaningful once a
> `future_frame_shift` model is trained.

---

## 6. Gotchas (explicit conditioning)

1. **Fully gated.** With `future_frame_shift=false` (the baseline action‑head config) the code path is
   byte‑for‑byte the original. The original launcher (`libero_training_wan22.sh`) and config are
   untouched.
2. **One masked chunk/clip** (`a_{-1}` at slot 0). 3 of 4 supervised at `max_chunk_size=4`; raise
   `max_chunk_size` (+ `num_frames=8·K+1`) to recover.
3. **Eval primes at every cache reset**, not just episode start — handled automatically via the
   `current_start_frame == 1 + num_frame_per_block` check; a single no‑op only at episode start.
4. **No future proprio.** State is paired with the action (`s_{b-1}`), which is the latest *real*
   proprio at decision time — train/eval consistent.
5. **Full FT is multi‑GPU.** Single‑GPU full fine‑tune OOMs at the optimizer step (ZeRO‑2 sharding).
6. Everything else (data, resolution, two‑env eval, OSMesa rendering, resume, wandb sibling‑run) is
   identical to `docs/LIBERO.md` — defer to it.

---

## 7. Reproduction checklist

```text
[ ] envs dreamzero + dreamzero_libero, ~/.libero/config.yaml, .env (HF_TOKEN + WANDB_API_KEY)   # LIBERO.md §10.1
[ ] download Wan2.2-5B + Wan2.1 CLIP + umt5 tokenizer; download + convert LIBERO dataset          # LIBERO.md §10.2
[ ] train: bash scripts/train/libero_training_wan22_future_shift.sh  (NUM_GPUS>=2, TRAIN_ARCH=full)# §3
[ ] eval watcher: scripts/eval/watch_eval_libero.sh on the future_shift OUTPUT_DIR (auto-primes)   # §4.1
[ ] or standalone: serve_dreamzero_libero.py (auto-detect) + run_libero_eval.py                    # §4.2
```
