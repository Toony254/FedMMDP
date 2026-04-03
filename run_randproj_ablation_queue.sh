#!/usr/bin/env bash
set -euo pipefail
cd /home/bd/data/zs/FedMMDP-base
for ABLATION in full with_rmg wo_cluster k1; do
  GPU=2 \
  SEED=3407 \
  DATASET=imagenet \
  MODEL=clip \
  DATA_ROOT=/home/bd/data/zs/FedMMDP/preprocessed_imagenet/domain_datasets \
  FEATURE_DIM=1024 \
  USE_PRETRAINED_PROJ=0 \
  COMM_ROUNDS=20 \
  ABLATION="$ABLATION" \
  bash ./run_fedmmdp_ablation_case.sh
done