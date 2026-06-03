"""
Modal deployment of the DreamZero websocket policy server on 2x H100.

Workflow:

    # 1) one-time: pull the HF checkpoint into the persistent Modal Volume
    modal run modal_server.py::download_checkpoint

    # 2) start the server. `modal serve` keeps the URL alive while the terminal
    #    is open; use `modal deploy` for a long-lived deployment.
    modal serve modal_server.py

    # Modal will print a URL like
    #   https://<workspace>--dreamzero-serve.modal.run
    # Use the host portion (no scheme) with port 443; the policy_client tries
    # ws:// first and falls back to wss://, so port 443 over TLS works.

    # 3) from sim-evals, point the eval at the Modal endpoint:
    python eval_utils/run_sim_eval.py \\
        --host <workspace>--dreamzero-serve.modal.run \\
        --port 443
"""

from pathlib import Path

import modal

APP_NAME = "dreamzero"
HF_REPO = "GEAR-Dreams/DreamZero-DROID"
CHECKPOINT_DIR_IN_VOL = "/checkpoints/DreamZero-DROID"
SERVER_PORT = 8000
HERE = Path(__file__).parent.resolve()

ckpt_vol = modal.Volume.from_name("dreamzero-checkpoints", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

# CUDA 12.9 base + PyTorch 2.8 wheels + flash-attn + TransformerEngine, then
# the dreamzero source installed editable so `groot.*` imports resolve.
image = (
    modal.Image.from_registry(
        # cudnn-devel variant bundles cuDNN headers — Transformer Engine builds from source need them
        "nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "build-essential",
        "clang",  # evdev's setup.py invokes clang directly
        "ninja-build",
        "curl",
        "ca-certificates",
    )
    .pip_install("packaging", "wheel", "setuptools>=67")
    .add_local_dir(
        HERE,
        "/root/dreamzero",
        copy=True,
        ignore=[
            "**/__pycache__",
            "**/.venv",
            "**/venv",
            "**/.git",
            "**/.idea",
            "**/.vscode",
            "**/*.pyc",
            "runs/**",
            "data/**",
            "checkpoints/**",
            "debug_image/**",
        ],
    )
    .workdir("/root/dreamzero")
    .run_commands(
        # Project deps, including the dreamzero package itself (so `groot.*` imports work).
        "pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129",
        # flash-attn must see torch during build.
        "MAX_JOBS=4 pip install --no-build-isolation flash-attn",
        # Skip Transformer Engine — its 2.10 cu12 wheel ships a partial cuDNN
        # that shadows torch's, breaking conv2d (CUDNN_STATUS_SUBLIBRARY_LOADING_
        # FAILED), and the 2.15+ API doesn't match dreamzero's call site.
        # socket_test_optimized_AR.py now sets ATTENTION_BACKEND=FA2 instead.
        gpu="H100",  # need a GPU at build time to compile flash-attn
    )
    # Keep these AFTER the heavy compile layer so editing them doesn't invalidate the TE cache.
    .apt_install("libgl1", "libglib2.0-0")  # opencv-python runtime deps
    .pip_install("hf_transfer")
    .env(
        {
            "HF_HOME": "/checkpoints/.hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "TORCH_CUDA_ARCH_LIST": "9.0+PTX",  # Hopper
        }
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={"/checkpoints": ckpt_vol},
    secrets=[hf_secret],
    timeout=60 * 60 * 2,
)
def download_checkpoint():
    """Pull the DreamZero-DROID weights into the persistent volume (run once)."""
    import os

    from huggingface_hub import snapshot_download

    os.makedirs(CHECKPOINT_DIR_IN_VOL, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="model",
        local_dir=CHECKPOINT_DIR_IN_VOL,
        max_workers=8,
    )
    ckpt_vol.commit()
    print(f"Downloaded {HF_REPO} -> {CHECKPOINT_DIR_IN_VOL}")


@app.function(
    gpu="H100:2",
    volumes={"/checkpoints": ckpt_vol},
    secrets=[hf_secret],
    timeout=60 * 60 * 6,
    max_containers=1,  # one process owns the 2 GPUs at a time
    scaledown_window=60 * 10,
)
@modal.web_server(port=SERVER_PORT, startup_timeout=60 * 15)
def serve():
    """Spawn torch.distributed across 2 H100s and the websocket policy server."""
    import subprocess

    subprocess.Popen(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "socket_test_optimized_AR.py",
            "--port",
            str(SERVER_PORT),
            "--enable-dit-cache",
            "--model-path",
            CHECKPOINT_DIR_IN_VOL,
        ],
        cwd="/root/dreamzero",
    )
