#!/usr/bin/env bash
set -euo pipefail

cd /home/bd/data/zs/FedMMDP-base

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
SEED="${SEED:-3407}"
COMM_ROUNDS="${COMM_ROUNDS:-20}"
USE_PRETRAINED_PROJ=1
DATASET=imagenet
MODEL=clip
DATA_ROOT=/home/bd/data/zs/FedMMDP/preprocessed_imagenet/domain_datasets
FEATURE_DIM=1024

launch_case() {
  local gpu="$1"
  local ablation="$2"
  setsid -f env \
    GPU="$gpu" \
    SEED="$SEED" \
    DATASET="$DATASET" \
    MODEL="$MODEL" \
    DATA_ROOT="$DATA_ROOT" \
    FEATURE_DIM="$FEATURE_DIM" \
    USE_PRETRAINED_PROJ="$USE_PRETRAINED_PROJ" \
    COMM_ROUNDS="$COMM_ROUNDS" \
    ABLATION="$ablation" \
    bash ./run_fedmmdp_ablation_case.sh >/dev/null 2>&1
}

launch_case "$GPU_A" full
launch_case "$GPU_A" with_rmg
launch_case "$GPU_A" wo_cluster
launch_case "$GPU_B" k1
launch_case "$GPU_B" inner3
