#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"
LOCAL_CONFIG="${LOCAL_CONFIG:-$PROJECT_ROOT/config/local_paths.sh}"
if [[ -f "$LOCAL_CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$LOCAL_CONFIG"
fi

CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-fedmmdp}"
USE_PRETRAINED_PROJ="${USE_PRETRAINED_PROJ:-0}"
PROJ_TAG=$([ "$USE_PRETRAINED_PROJ" -eq 1 ] && echo preproj || echo randproj)
COMM_ROUNDS="${COMM_ROUNDS:-50}"
PUB_DATA_NUM="${PUB_DATA_NUM:-5000}"
GPUS="${GPUS:-1,1,1,1,2,2,2,2,3,3,3}"

DATASET="imagenet"
DATA_ROOT="${DATA_ROOT:-${FEDMMDP_IMAGENET_ROOT:-data/imagenet/domain_datasets}}"
MODEL="clip"
FEATURE_DIM="1024"
FEDMMDP_NAME="${FEDMMDP_NAME:-FedMMDP-imagenet-clip-${PROJ_TAG}-noRMG}"
FEDMMDP_LOG="${FEDMMDP_LOG:-outputs/output_FedMMDP_imagenet_clip_${PROJ_TAG}_noRMG.log}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "[ERROR] conda binary not found: $CONDA_BIN" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] data root not found: $DATA_ROOT" >&2
  exit 1
fi

mkdir -p outputs

eval "$("$CONDA_BIN" shell.bash hook)"
if [[ -n "$ENV_NAME" ]]; then
  conda activate "$ENV_NAME"
fi

IFS=',' read -r -a GPU_SLOTS <<< "${GPUS// /}"
if (( ${#GPU_SLOTS[@]} < 11 )); then
  echo "[ERROR] GPUS must contain at least 11 comma-separated entries, got: $GPUS" >&2
  exit 1
fi

declare -a PIDS=()

launch_run() {
  local gpu="$1"
  local name="$2"
  local algo="$3"
  local log_file="$4"
  shift 4

  echo "[LAUNCH] gpu=$gpu algo=$algo name=$name log=$log_file"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python -u src/main.py \
    --name "$name" \
    --FL_algorithm "$algo" \
    --dataset "$DATASET" \
    --data_root "$DATA_ROOT" \
    --model "$MODEL" \
    --feature_dim "$FEATURE_DIM" \
    --lr 1e-5 \
    --local_epochs 1 \
    --comm_rounds "$COMM_ROUNDS" \
    --batch_size 256 \
    --use_pretrained_proj "$USE_PRETRAINED_PROJ" \
    --partition hetero \
    "$@" >"$log_file" 2>&1 &
  PIDS+=($!)
}

launch_run "${GPU_SLOTS[0]}" "FedMMDP-avg" "FedAvg" "outputs/output_avg_clip_${PROJ_TAG}.log"
launch_run "${GPU_SLOTS[1]}" "FedMMDP-prox" "FedProx" "outputs/output_prox_${PROJ_TAG}.log"
launch_run "${GPU_SLOTS[2]}" "FedMMDP-md" "FedMD" "outputs/output_md_${PROJ_TAG}.log" \
  --pub_data_num "$PUB_DATA_NUM"
launch_run "${GPU_SLOTS[3]}" "FedMMDP-df" "FedDF" "outputs/output_df_${PROJ_TAG}.log" \
  --pub_data_num "$PUB_DATA_NUM"
launch_run "${GPU_SLOTS[4]}" "FedMMDP-Cream" "Cream" "outputs/output_cream_${PROJ_TAG}.log" \
  --pub_data_num "$PUB_DATA_NUM"
launch_run "${GPU_SLOTS[5]}" "FedMMDP-Harmony" "Harmony" "outputs/output_Harmony_${PROJ_TAG}.log" \
  --harmony_stage1_rounds 10 \
  --harmony_clusters 2 \
  --harmony_cluster_mode fixed
launch_run "${GPU_SLOTS[6]}" "FedMMDP-MASA" "MASA" "outputs/output_MASA_${PROJ_TAG}.log"
launch_run "${GPU_SLOTS[7]}" "FedMMDP-MEMA" "FedMEMA" "outputs/output_FedMEMA_${PROJ_TAG}.log"
launch_run "${GPU_SLOTS[8]}" "FedMMDP-Mobile" "FedMobile" "outputs/output_FedMobile_${PROJ_TAG}.log"
launch_run "${GPU_SLOTS[9]}" "FedMMDP-MEKT" "FedMEKT" "outputs/output_FedMEKT_${PROJ_TAG}.log"
launch_run "${GPU_SLOTS[10]}" "$FEDMMDP_NAME" "FedMMDP" "$FEDMMDP_LOG" \
  --fedmmdp_log_aux_losses 1 \
  --fedmmdp_disable_rmg_loss 1

echo "[LAUNCH] data_root=$DATA_ROOT"
echo "[LAUNCH] proj=$PROJ_TAG comm_rounds=$COMM_ROUNDS"
echo "[LAUNCH] started ${#PIDS[@]} jobs"
echo "[LAUNCH] pids=${PIDS[*]}"
