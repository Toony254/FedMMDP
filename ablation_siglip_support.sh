#!/usr/bin/env bash
set -euo pipefail

cd /home/bd/data/zs/FedMMDP-base

GPU="${GPU:-0}"
USE_PRETRAINED_PROJ="${USE_PRETRAINED_PROJ:-1}"
COMM_ROUNDS="${COMM_ROUNDS:-30}"
CASE_LIST="${CASE_LIST:-full with_rmg wo_cluster}"

for ABLATION in $CASE_LIST; do
  DATASET=imagenet \
  MODEL=siglip \
  DATA_ROOT=/home/bd/data/zs/FedMMDP/preprocessed_imagenet/domain_datasets/siglip \
  FEATURE_DIM=768 \
  GPU="$GPU" \
  USE_PRETRAINED_PROJ="$USE_PRETRAINED_PROJ" \
  COMM_ROUNDS="$COMM_ROUNDS" \
  ABLATION="$ABLATION" \
  bash ./run_fedmmdp_ablation_case.sh

  DATASET=iapr \
  MODEL=siglip \
  DATA_ROOT=/home/bd/data/zs/FedMMDP/preprocessed_iapr_siglip/domain_datasets \
  FEATURE_DIM=768 \
  GPU="$GPU" \
  USE_PRETRAINED_PROJ="$USE_PRETRAINED_PROJ" \
  COMM_ROUNDS="$COMM_ROUNDS" \
  ABLATION="$ABLATION" \
  bash ./run_fedmmdp_ablation_case.sh
done
