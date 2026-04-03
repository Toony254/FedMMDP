#!/usr/bin/env bash
set -euo pipefail

cd /home/bd/data/zs/FedMMDP-base

GPU=3 \
DATASET=iapr \
MODEL=clip \
DATA_ROOT=/home/bd/data/zs/FedMMDP/preprocessed_iapr/domain_datasets \
FEATURE_DIM=1024 \
USE_PRETRAINED_PROJ=1 \
COMM_ROUNDS="${COMM_ROUNDS:-2}" \
ABLATION="${ABLATION:-wo_cluster}" \
LOG_AUX=1 \
bash ./run_fedmmdp_ablation_case.sh
