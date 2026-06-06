#!/bin/bash
# DreamZero LIBERO training with the Wan2.2-TI2V-5B backbone (full or LoRA finetune).
#
# Usage:
#   bash scripts/train/libero_training_wan22.sh
#
# Prerequisites:
#   - LIBERO dataset converted to DreamZero/GEAR LeRobot format at LIBERO_DATA_ROOT
#       python scripts/data/convert_libero_to_dreamzero.py --src data/libero_raw_lerobot --dst data/libero_lerobot
#   - Wan2.2-TI2V-5B weights at WAN22_CKPT_DIR
#       hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
#   - CLIP image encoder from Wan2.1 (Wan2.2-TI2V-5B does not ship it)
#       hf download Wan-AI/Wan2.1-I2V-14B-480P --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#           --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
#   - umt5-xxl tokenizer files at TOKENIZER_DIR (bundled inside Wan2.2-TI2V-5B/google/umt5-xxl)

export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -n "$DREAMZERO_ROOT" ] && [ -d "$DREAMZERO_ROOT/groot" ]; then
    : # keep existing and valid
elif [ -d "$SCRIPT_REPO_ROOT/groot" ]; then
    DREAMZERO_ROOT="$SCRIPT_REPO_ROOT"
else
    DREAMZERO_ROOT="${DREAMZERO_ROOT:-$SCRIPT_REPO_ROOT}"
fi
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: No groot/ under $DREAMZERO_ROOT. Set DREAMZERO_ROOT to the dreamzero repo root."
    exit 1
fi

# ============ USER CONFIGURATION ============
NUM_GPUS=${NUM_GPUS:-8}
LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-"$DREAMZERO_ROOT/data/libero_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_libero_wan22"}

# Training scale: per-GPU batch size, max steps, and architecture (full|lora).
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
MAX_STEPS=${MAX_STEPS:-100}
TRAIN_ARCH=${TRAIN_ARCH:-full}          # 'full' (no LoRA) per the request; set 'lora' for LoRA
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_LORA_ONLY=${SAVE_LORA_ONLY:-false}  # only relevant if TRAIN_ARCH=lora
# DeepSpeed config. Default ZeRO-2 (shards optimizer across GPUs). For single-GPU full fine-tune
# use groot/vla/configs/deepspeed/zero2_offload.json (offloads the optimizer to CPU RAM).
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-groot/vla/configs/deepspeed/zero2.json}

# Wan2.2-TI2V-5B checkpoint (diffusion weights, T5 encoder, VAE)
WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B"}
# CLIP image encoder (from Wan2.1; Wan2.2-TI2V-5B does not include it)
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
# umt5-xxl tokenizer (bundled inside the Wan2.2 repo)
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/umt5-xxl"}
# =============================================

if [ ! -d "$LIBERO_DATA_ROOT" ]; then
    echo "ERROR: LIBERO dataset not found at $LIBERO_DATA_ROOT"
    echo "Convert it with: python scripts/data/convert_libero_to_dreamzero.py --src data/libero_raw_lerobot --dst $LIBERO_DATA_ROOT"
    exit 1
fi
if [ ! -f "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "ERROR: CLIP image encoder not found in $IMAGE_ENCODER_DIR"
    exit 1
fi

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_CMD=( "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
echo "Using python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
cd "$DREAMZERO_ROOT"

"${RUN_CMD[@]}" \
    report_to=${REPORT_TO:-wandb} \
    data=dreamzero/libero_relative_wan22 \
    wandb_project=dreamzero_libero \
    train_architecture=$TRAIN_ARCH \
    num_frames=33 \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=${LR:-1e-5} \
    training_args.deepspeed="$DEEPSPEED_CONFIG" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_BATCH_SIZE \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=${NUM_WORKERS:-1} \
    save_lora_only=$SAVE_LORA_ONLY \
    max_chunk_size=4 \
    save_strategy=$SAVE_STRATEGY \
    libero_data_root=$LIBERO_DATA_ROOT \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
