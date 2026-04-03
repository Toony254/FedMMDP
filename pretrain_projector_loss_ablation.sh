#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${1:-clip}"
PROJECTOR="${PROJECTOR:-mlp+norm}"
SAVE_PATH="${SAVE_PATH:-saved/projector_weights}"
LOG_DIR="${LOG_DIR:-outputs}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
TEMPERATURE="${TEMPERATURE:-0.07}"
MARGIN="${MARGIN:-0.2}"
COCO_ROOT="${COCO_ROOT:-/home/bd/data/zs/data/COCO}"
FLICKR_SPLIT="${FLICKR_SPLIT:-dataset_k_split.pkl}"

mkdir -p "$SAVE_PATH" "$LOG_DIR"

if [[ "$MODEL" == "clip" ]]; then
  TRAIN_SCRIPT="src/networks/train_projector.py"
  EXTRA_ARGS=(--backbone "${BACKBONE:-RN50}")
elif [[ "$MODEL" == "siglip" ]]; then
  TRAIN_SCRIPT="src/networks/train_projector_SigLIP.py"
  EXTRA_ARGS=()
else
  echo "Unsupported model: $MODEL"
  exit 1
fi

COMMON_ARGS=(
  --save_path "$SAVE_PATH"
  --lr "$LR"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --temperature "$TEMPERATURE"
  --margin "$MARGIN"
  --projector "$PROJECTOR"
  --coco_json "$COCO_ROOT/annotations/captions_train2017.json"
  --coco_img_dir "$COCO_ROOT/train2017"
  --coco_val_json "$COCO_ROOT/annotations/captions_val2017.json"
  --coco_val_img_dir "$COCO_ROOT/val2017"
  --flickr_split "$FLICKR_SPLIT"
)

LOSS_MODES=("cl_only" "rmg_only" "max_margin")

for loss_mode in "${LOSS_MODES[@]}"; do
  log_file="$LOG_DIR/pretrain_${MODEL}_${PROJECTOR}_${loss_mode}.log"
  echo "[launch] model=$MODEL projector=$PROJECTOR loss_mode=$loss_mode log=$log_file"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --loss_mode "$loss_mode" \
    >"$log_file" 2>&1
done
