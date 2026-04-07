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
GPU="${GPU:-0}"
SEED="${SEED:-3407}"
DATASET="${DATASET:?DATASET is required}"
MODEL="${MODEL:?MODEL is required}"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required}"
ABLATION="${ABLATION:-full}"
USE_PRETRAINED_PROJ="${USE_PRETRAINED_PROJ:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-5}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
COMM_ROUNDS="${COMM_ROUNDS:-30}"
PARTITION="${PARTITION:-hetero}"
FEATURE_DIM="${FEATURE_DIM:-1024}"
PUB_DATA_NUM="${PUB_DATA_NUM:-5000}"
N_CLUSTERS="${N_CLUSTERS:-5}"
TAU="${TAU:-0.5}"
CLUSTER_WEIGHT="${CLUSTER_WEIGHT:-1.0}"
RMG_WEIGHT="${RMG_WEIGHT:-1.0}"
INNER_STEPS="${INNER_STEPS:-1}"
LOG_AUX="${LOG_AUX:-1}"
AUX_NORM_EPS="${AUX_NORM_EPS:-1e-6}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

PROJ_TAG=$([ "$USE_PRETRAINED_PROJ" -eq 1 ] && echo preproj || echo randproj)

COMMON_ARGS=(
  --FL_algorithm FedMMDP
  --dataset "$DATASET"
  --data_root "$DATA_ROOT"
  --model "$MODEL"
  --seed "$SEED"
  --feature_dim "$FEATURE_DIM"
  --lr "$LR"
  --local_epochs "$LOCAL_EPOCHS"
  --comm_rounds "$COMM_ROUNDS"
  --batch_size "$BATCH_SIZE"
  --use_pretrained_proj "$USE_PRETRAINED_PROJ"
  --partition "$PARTITION"
  --n_clusters "$N_CLUSTERS"
  --tau "$TAU"
  --cluster_weight "$CLUSTER_WEIGHT"
  --rmg_weight "$RMG_WEIGHT"
  --fedmmdp_cluster_inner_steps "$INNER_STEPS"
  --fedmmdp_log_aux_losses "$LOG_AUX"
  --fedmmdp_aux_norm_eps "$AUX_NORM_EPS"
)

ABLATION_ARGS=()
case "$ABLATION" in
  full)
    ABLATION_TAG="full"
    ABLATION_ARGS+=(--fedmmdp_disable_rmg_loss 1 --rmg_weight 0)
    ;;
  with_rmg)
    ABLATION_TAG="withRMG"
    ;;
  wo_cluster)
    ABLATION_TAG="woCluster"
    ABLATION_ARGS+=(--fedmmdp_disable_cluster_loss 1 --fedmmdp_disable_rmg_loss 1 --cluster_weight 0 --rmg_weight 0)
    ;;
  wo_rmg)
    ABLATION_TAG="woRMG"
    ABLATION_ARGS+=(--fedmmdp_disable_rmg_loss 1 --rmg_weight 0)
    ;;
  wo_both)
    ABLATION_TAG="woBoth"
    ABLATION_ARGS+=(--fedmmdp_disable_cluster_loss 1 --fedmmdp_disable_rmg_loss 1 --cluster_weight 0 --rmg_weight 0)
    ;;
  k1)
    ABLATION_TAG="k1"
    ABLATION_ARGS+=(--n_clusters 1)
    ;;
  inner3)
    ABLATION_TAG="inner3"
    ABLATION_ARGS+=(--fedmmdp_cluster_inner_steps 3)
    ;;
  *)
    echo "Unsupported ABLATION=$ABLATION"
    exit 1
    ;;
esac

if [ "$MODEL" = "siglip" ]; then
  NAME="FedMMDP-ABL-${DATASET}-siglip-${PROJ_TAG}-${ABLATION_TAG}-s${SEED}"
else
  NAME="FedMMDP-ABL-${DATASET}-clip-${PROJ_TAG}-${ABLATION_TAG}-s${SEED}"
fi

LOG_FILE="outputs/output_${NAME}.log"

if [ "$DATASET" = "imagenet" ] || [ "$DATASET" = "iapr" ]; then
  :
else
  echo "Unsupported DATASET=$DATASET for ablation script"
  exit 1
fi

if [ "$MODEL" = "siglip" ]; then
  :
elif [ "$MODEL" = "clip" ]; then
  :
else
  echo "Unsupported MODEL=$MODEL for ablation script"
  exit 1
fi

if [[ "$ABLATION" == "full" || "$ABLATION" == "with_rmg" || "$ABLATION" == "wo_cluster" || "$ABLATION" == "wo_rmg" || "$ABLATION" == "wo_both" ]]; then
  :
elif [[ "$ABLATION" == "k1" || "$ABLATION" == "inner3" ]]; then
  :
fi

if [ "$MODEL" = "clip" ] || [ "$MODEL" = "siglip" ]; then
  EXTRA_DATA_ARGS=()
else
  EXTRA_DATA_ARGS=(--pub_data_num "$PUB_DATA_NUM")
fi

EXTRA_ARGS_ARR=()
if [ -n "$EXTRA_ARGS" ]; then
  read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"
fi

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

echo "[ABLA] gpu=$GPU dataset=$DATASET model=$MODEL proj=$PROJ_TAG ablation=$ABLATION rounds=$COMM_ROUNDS name=$NAME" | tee -a "$LOG_FILE"
echo "[ABLA] log_file=$LOG_FILE start_time=$(date '+%F %T')" | tee -a "$LOG_FILE"

eval "$("$CONDA_BIN" shell.bash hook)"
if [[ -n "$ENV_NAME" ]]; then
  conda activate "$ENV_NAME"
fi

set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 python -u src/main.py \
  --name "$NAME" \
  --fedmmdp_ablation_tag "$ABLATION_TAG" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_DATA_ARGS[@]}" \
  "${ABLATION_ARGS[@]}" \
  "${EXTRA_ARGS_ARR[@]}" \
  2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e

echo "[ABLA] finish_time=$(date '+%F %T') exit_status=$status" | tee -a "$LOG_FILE"
exit "$status"
