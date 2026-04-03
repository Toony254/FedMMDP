#!/usr/bin/env bash
set -euo pipefail

cd /home/bd/data/zs/FedMMDP-base

GPU="${GPU:-0}"
USE_PRETRAINED_PROJ="${USE_PRETRAINED_PROJ:-1}"
COMM_ROUNDS="${COMM_ROUNDS:-30}"
CASE_LIST="${CASE_LIST:-full with_rmg wo_cluster k1 inner3}"

for ABLATION in $CASE_LIST; do
  DATASET=imagenet \
  MODEL=clip \
  DATA_ROOT=/home/bd/data/zs/FedMMDP/preprocessed_imagenet/domain_datasets \
  FEATURE_DIM=1024 \
  GPU="$GPU" \
  USE_PRETRAINED_PROJ="$USE_PRETRAINED_PROJ" \
  COMM_ROUNDS="$COMM_ROUNDS" \
  ABLATION="$ABLATION" \
  bash ./run_fedmmdp_ablation_case.sh

  DATASET=iapr \
  MODEL=clip \
  DATA_ROOT=/home/bd/data/zs/FedMMDP/preprocessed_iapr/domain_datasets \
  FEATURE_DIM=1024 \
  GPU="$GPU" \
  USE_PRETRAINED_PROJ="$USE_PRETRAINED_PROJ" \
  COMM_ROUNDS="$COMM_ROUNDS" \
  ABLATION="$ABLATION" \
  bash ./run_fedmmdp_ablation_case.sh
done
