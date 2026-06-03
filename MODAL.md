# Self-hosting DreamZero on Modal + running sim-evals against it

This directory contains a Modal deployment (`modal_server.py`) that runs the
DreamZero-DROID websocket policy server on 2x H100s. A local sim-evals
checkout drives the eval and connects to the Modal endpoint over `wss://`.

## What gets deployed

- **Image**: `nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04` + Python 3.11 + the
  dreamzero pyproject deps (torch 2.8 / cu129) + flash-attn 2.
- **Volume** `dreamzero-checkpoints`: holds the
  [`GEAR-Dreams/DreamZero-DROID`](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)
  ~28 GB checkpoint between runs.
- **Function** `serve`: 2x H100, exposed via `@modal.web_server(port=8000)` as
  `https://<workspace>--dreamzero-serve.modal.run`. Launches
  `socket_test_optimized_AR.py` via `torch.distributed.run --nproc_per_node=2`.

## Prerequisites

1. **Modal account + CLI** — `pip install modal && modal token new`.
2. **Hugging Face token** with read access to `GEAR-Dreams/DreamZero-DROID`.
   Save locally (`hf auth login`) and upload to Modal as a secret named
   `huggingface`:

   ```bash
   HF_TOKEN_VAL="$(cat ~/.cache/huggingface/token)"
   modal secret create huggingface \
       HF_TOKEN="$HF_TOKEN_VAL" \
       HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL"
   ```

3. **sim-evals checkout** with its venv and assets:

   ```bash
   cd /path/to/sim-evals
   uv sync
   source .venv/bin/activate
   uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets
   ```

   The first `import isaaclab` will prompt for the Omniverse EULA — accept it
   once (pipe `Yes` into a one-off `python -c "import isaaclab"` if needed).

## One-time setup: pull the checkpoint into the Volume

```bash
cd /path/to/dreamzero
modal run modal_server.py::download_checkpoint
```

This triggers the first image build (~15 min — flash-attn compile) and then
`snapshot_download`s the model into the persistent volume. Subsequent deploys
reuse the build cache and the volume.

## Deploy the server

```bash
modal deploy modal_server.py
```

This prints the public URL, e.g. `https://<workspace>--dreamzero-serve.modal.run`.
The container is lazy — it spins up on the first request.

## Warm up before evaluating

The eval client (`policy_client.py:42-48`) calls `websockets.sync.client.connect()`
without an `open_timeout`, so it defaults to ~10 s. That's far shorter than
Modal's cold-start (we routinely see 200–450 s while the 14 B checkpoint loads
into VRAM across 2 GPUs). Hold the connection from outside first so the eval
hits a warm container:

```bash
python -c "
import websockets.sync.client, time
url = 'wss://<workspace>--dreamzero-serve.modal.run:443'
start = time.time()
conn = websockets.sync.client.connect(url, compression=None, max_size=None, open_timeout=900)
print(f'[{time.time()-start:.1f}s] Handshake OK')
conn.recv(); conn.close()
"
```

The container then stays warm for the duration of `scaledown_window` (set to
10 min in `modal_server.py`).

## Run the eval

From the sim-evals venv, point the eval at the Modal URL. Note: pass the host
*without* the scheme, with port `443` (the client tries `ws://` first and
falls back to `wss://`, which is what Modal serves).

Single scene:

```bash
cd /path/to/sim-evals
source .venv/bin/activate
python /path/to/dreamzero/eval_utils/run_sim_eval.py \
    --host <workspace>--dreamzero-serve.modal.run \
    --port 443 \
    --episodes 10 \
    --scene 1 \
    --headless
```

All 3 scenes, N episodes each:

```bash
for s in 1 2 3; do
  python /path/to/dreamzero/eval_utils/run_sim_eval.py \
      --host <workspace>--dreamzero-serve.modal.run \
      --port 443 \
      --episodes 10 --scene $s --headless
done
```

Videos land in `sim-evals/runs/<date>/<HH-MM-SS>/episode_<i>.mp4`. Each scene
invocation creates its own timestamped dir.

Scenes:

| Scene | Instruction |
|---|---|
| 1 | put the cube in the bowl |
| 2 | pick up the can and put it in the mug |
| 3 | put the banana in the bin |

## Tear down to stop billing

`@modal.web_server` containers idle out after `scaledown_window` (10 min),
but you can force-stop immediately:

```bash
modal app stop dreamzero -y
```

The image and Volume are persistent and free; only running containers cost
money. To re-run later, redeploy and warm up — the build cache and checkpoint
are reused.

## Footguns we hit (and how the file avoids them)

- **`evdev` wheel fails to build** because the python-build-standalone image
  has `CC=clang` baked in. → `apt_install("clang")`.
- **Transformer Engine** is brittle here: TE ≥ 2.15 added `NVTE_QKV_Format`
  args that `groot/vla/model/dreamzero/modules/cudnn_attention.py:319`
  doesn't pass, and TE 2.10's `transformer_engine_cu12` wheel ships a partial
  cuDNN that shadows torch's and crashes conv2d with
  `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`. → We don't install TE at all and
  patch `socket_test_optimized_AR.py:745` to set
  `ATTENTION_BACKEND=FA2`. `wan2_1_attention.py:233-235` already falls back
  to FA2 cleanly when TE is absent.
- **opencv import fails** with `libGL.so.1: cannot open shared object file`. →
  `apt_install("libgl1", "libglib2.0-0")` as a late layer.
- **`HF_HUB_ENABLE_HF_TRANSFER=1` requires `hf_transfer`** — pip-installed in
  a late layer so it doesn't invalidate the heavy compile cache.
- **Modal's web_server edge times out HTTP requests at ~150 s** with a 303,
  even though `startup_timeout` is 15 min. Don't use `curl /healthz` as a
  readiness probe; use the websocket warm-up snippet above. Also: the
  `_health_check` callback at `socket_test_optimized_AR.py:733` is never
  wired into the actual `RoboarenaServer` in `eval_utils/policy_server.py`,
  so `/healthz` will get a 426 even when the server is ready.
- **Source-file edits invalidate the heavy compile cache** because
  `add_local_dir` sits before the `run_commands` step in `modal_server.py`.
  If you iterate often, restructure: install deps via
  `pip_install_from_pyproject`, then add the source as a late layer, then
  `pip install -e . --no-deps`.
