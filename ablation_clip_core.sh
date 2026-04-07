#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"
LOCAL_CONFIG="${LOCAL_CONFIG:-$PROJECT_ROOT/config/local_paths.sh}"
if [[ -f "$LOCAL_CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$LOCAL_CONFIG"
fi

GPU="${GPU:-0}"
USE_PRETRAINED_PROJ="${USE_PRETRAINED_PROJ:-1}"
COMM_ROUNDS="${COMM_ROUNDS:-30}"
CASE_LIST="${CASE_LIST:-full with_rmg wo_cluster k1 inner3}"
IMAGENET_ROOT="${IMAGENET_ROOT:-${FEDMMDP_IMAGENET_ROOT:-data/imagenet/domain_datasets}}"
IAPR_ROOT="${IAPR_ROOT:-${FEDMMDP_IAPR_ROOT:-data/iapr/domain_datasets}}"

for ABLATION in $CASE_LIST; do
  DATASET=imagenet \
  MODEL=clip \
  DATA_ROOT="$IMAGENET_ROOT" \
  FEATURE_DIM=1024 \
  GPU="$GPU" \
  USE_PRETRAINED_PROJ="$USE_PRETRAINED_PROJ" \
  COMM_ROUNDS="$COMM_ROUNDS" \
  ABLATION="$ABLATION" \
  bash ./run_fedmmdp_ablation_case.sh

  DATASET=iapr \
  MODEL=clip \
  DATA_ROOT="$IAPR_ROOT" \
  FEATURE_DIM=1024 \
  GPU="$GPU" \
  USE_PRETRAINED_PROJ="$USE_PRETRAINED_PROJ" \
  COMM_ROUNDS="$COMM_ROUNDS" \
  ABLATION="$ABLATION" \
  bash ./run_fedmmdp_ablation_case.sh
done
