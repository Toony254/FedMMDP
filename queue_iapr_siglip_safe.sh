#!/usr/bin/env bash
set -euo pipefail

cd /home/bd/data/zs/FedMMDP-base

CONDA_BIN="/mnt/data/software/anaconda/bin/conda"
ENV_NAME="zs_vita"
DATA_ROOT="/home/bd/data/zs/FedMMDP/preprocessed_iapr_siglip/domain_datasets"
POLL_SECONDS="${POLL_SECONDS:-120}"
SCHEDULE_INTERVAL_SECONDS="${SCHEDULE_INTERVAL_SECONDS:-8}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-80}"
MAX_NEW_JOBS_PER_GPU="${MAX_NEW_JOBS_PER_GPU:-4}"
JOB_EST_MEM_MB="${JOB_EST_MEM_MB:-3000}"
RESERVE_MEM_MB="${RESERVE_MEM_MB:-5000}"
QUEUE_LOG="outputs/queue_iapr_siglip_safe.log"
STATE_DIR="outputs/queue_iapr_siglip_state"
NEXT_FILE="$STATE_DIR/next_index"
LOCK_FILE="$STATE_DIR/claim.lock"

mkdir -p "$(dirname "$QUEUE_LOG")" "$STATE_DIR"
: > "$QUEUE_LOG"
echo 0 > "$NEXT_FILE"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$QUEUE_LOG" >&2
}

job_done_marker() {
  local tag="$1"
  printf '%s/%s.done' "$STATE_DIR" "$tag"
}

job_fail_marker() {
  local tag="$1"
  printf '%s/%s.failed' "$STATE_DIR" "$tag"
}

job_slot_marker() {
  local tag="$1"
  local gpu="$2"
  printf '%s/%s.gpu%s.slot' "$STATE_DIR" "$tag" "$gpu"
}

launch_job() {
  local tag="$1"
  local logfile="$2"
  local gpu="$3"
  local cmd="$4"

  local done_marker
  local fail_marker
  local slot_marker
  done_marker="$(job_done_marker "$tag")"
  fail_marker="$(job_fail_marker "$tag")"
  slot_marker="$(job_slot_marker "$tag" "$gpu")"

  rm -f "$fail_marker"
  : > "$slot_marker"
  log "launch $tag on gpu=$gpu"
  nohup bash -lc "
    cd '/home/bd/data/zs/FedMMDP-base'
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTHONUNBUFFERED=1
    '$CONDA_BIN' run -n '$ENV_NAME' python -u src/main.py $cmd > '$logfile' 2>&1
    status=\$?
    rm -f '$slot_marker'
    if [ \$status -eq 0 ]; then
      touch '$done_marker'
    else
      echo \$status > '$fail_marker'
    fi
    exit \$status
  " >/dev/null 2>&1 &
}

declare -a JOBS=(
  "FedMMDP_preproj|outputs/output_FedMMDP_siglip_iapr_preproj.log|--name FedMMDP-siglip --FL_algorithm FedMMDP --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 1 --partition hetero"
  "Harmony_preproj|outputs/output_Harmony_siglip_iapr_preproj.log|--name FedMMDP-Harmony-siglip --FL_algorithm Harmony --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --harmony_stage1_rounds 10 --harmony_clusters 2 --harmony_cluster_mode fixed --model siglip --batch_size 256 --use_pretrained_proj 1 --partition hetero"
  "FedMEKT_preproj|outputs/output_FedMEKT_siglip_iapr_preproj.log|--name FedMMDP-MEKT-siglip --FL_algorithm FedMEKT --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 1 --partition hetero"
  "FedMobile_preproj|outputs/output_FedMobile_siglip_iapr_preproj.log|--name FedMMDP-Mobile-siglip --FL_algorithm FedMobile --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 1 --partition hetero"
  "Cream_preproj|outputs/output_cream_siglip_iapr_preproj.log|--name FedMMDP-Cream-siglip --FL_algorithm Cream --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --pub_data_num 5000 --use_pretrained_proj 1 --partition hetero"
  "MASA_preproj|outputs/output_MASA_siglip_iapr_preproj.log|--name FedMMDP-MASA-siglip --FL_algorithm MASA --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 1 --partition hetero"
  "FedMD_preproj|outputs/output_md_siglip_iapr_preproj.log|--name FedMMDP-md-siglip --FL_algorithm FedMD --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --pub_data_num 5000 --use_pretrained_proj 1 --partition hetero"
  "FedDF_preproj|outputs/output_df_siglip_iapr_preproj.log|--name FedMMDP-df-siglip --FL_algorithm FedDF --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --pub_data_num 5000 --use_pretrained_proj 1 --partition hetero"
  "FedAvg_preproj|outputs/output_avg_siglip_iapr_preproj.log|--name FedMMDP-avg-siglip --FL_algorithm FedAvg --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 1 --partition hetero"
  "FedProx_preproj|outputs/output_prox_siglip_iapr_preproj.log|--name FedMMDP-prox-siglip --FL_algorithm FedProx --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 1 --partition hetero"
  "FedMEMA_preproj|outputs/output_FedMEMA_siglip_iapr_preproj.log|--name FedMMDP-MEMA-siglip --FL_algorithm FedMEMA --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 1 --partition hetero"
  "FedMMDP_randproj|outputs/output_FedMMDP_siglip_iapr_randproj.log|--name FedMMDP-siglip --FL_algorithm FedMMDP --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 0 --partition hetero"
  "Harmony_randproj|outputs/output_Harmony_siglip_iapr_randproj.log|--name FedMMDP-Harmony-siglip --FL_algorithm Harmony --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --harmony_stage1_rounds 10 --harmony_clusters 2 --harmony_cluster_mode fixed --model siglip --batch_size 256 --use_pretrained_proj 0 --partition hetero"
  "FedMEKT_randproj|outputs/output_FedMEKT_siglip_iapr_randproj.log|--name FedMMDP-MEKT-siglip --FL_algorithm FedMEKT --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 0 --partition hetero"
  "FedMobile_randproj|outputs/output_FedMobile_siglip_iapr_randproj.log|--name FedMMDP-Mobile-siglip --FL_algorithm FedMobile --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 0 --partition hetero"
  "Cream_randproj|outputs/output_cream_siglip_iapr_randproj.log|--name FedMMDP-Cream-siglip --FL_algorithm Cream --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --pub_data_num 5000 --use_pretrained_proj 0 --partition hetero"
  "MASA_randproj|outputs/output_MASA_siglip_iapr_randproj.log|--name FedMMDP-MASA-siglip --FL_algorithm MASA --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 0 --partition hetero"
  "FedMD_randproj|outputs/output_md_siglip_iapr_randproj.log|--name FedMMDP-md-siglip --FL_algorithm FedMD --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --pub_data_num 5000 --use_pretrained_proj 0 --partition hetero"
  "FedDF_randproj|outputs/output_df_siglip_iapr_randproj.log|--name FedMMDP-df-siglip --FL_algorithm FedDF --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --pub_data_num 5000 --use_pretrained_proj 0 --partition hetero"
  "FedAvg_randproj|outputs/output_avg_siglip_iapr_randproj.log|--name FedMMDP-avg-siglip --FL_algorithm FedAvg --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 0 --partition hetero"
  "FedProx_randproj|outputs/output_prox_siglip_iapr_randproj.log|--name FedMMDP-prox-siglip --FL_algorithm FedProx --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 0 --partition hetero"
  "FedMEMA_randproj|outputs/output_FedMEMA_siglip_iapr_randproj.log|--name FedMMDP-MEMA-siglip --FL_algorithm FedMEMA --dataset iapr --data_root $DATA_ROOT --feature_dim 768 --lr 1e-5 --local_epochs 1 --comm_rounds 50 --model siglip --batch_size 256 --use_pretrained_proj 0 --partition hetero"
)

claim_next_job() {
  exec 9>>"$LOCK_FILE"
  flock 9
  local idx
  idx="$(cat "$NEXT_FILE")"
  while [ "$idx" -lt "${#JOBS[@]}" ]; do
    local job="${JOBS[$idx]}"
    IFS='|' read -r tag _ _ <<< "$job"
    if [ -f "$(job_done_marker "$tag")" ]; then
      idx=$((idx + 1))
      continue
    fi
    echo $((idx + 1)) > "$NEXT_FILE"
    flock -u 9
    exec 9>&-
    printf '%s\n' "$job"
    return 0
  done
  echo "$idx" > "$NEXT_FILE"
  flock -u 9
  exec 9>&-
  return 1
}

jobs_remaining() {
  local idx
  idx="$(cat "$NEXT_FILE")"
  [ "$idx" -lt "${#JOBS[@]}" ]
}

select_gpu() {
  DATA_ROOT="$DATA_ROOT" \
  STATE_DIR="$STATE_DIR" \
  MAX_GPU_UTIL="$MAX_GPU_UTIL" \
  MAX_NEW_JOBS_PER_GPU="$MAX_NEW_JOBS_PER_GPU" \
  JOB_EST_MEM_MB="$JOB_EST_MEM_MB" \
  RESERVE_MEM_MB="$RESERVE_MEM_MB" \
  python3 - <<'PY'
import glob
import os
import subprocess
import sys

state_dir = os.environ["STATE_DIR"]
max_util = int(os.environ["MAX_GPU_UTIL"])
max_jobs = int(os.environ["MAX_NEW_JOBS_PER_GPU"])
job_est = int(os.environ["JOB_EST_MEM_MB"])
reserve = int(os.environ["RESERVE_MEM_MB"])

slot_counts = {}
for path in glob.glob(os.path.join(state_dir, "*.gpu*.slot")):
    base = os.path.basename(path)
    if ".gpu" not in base:
        continue
    gpu = base.split(".gpu", 1)[1].split(".slot", 1)[0]
    slot_counts[gpu] = slot_counts.get(gpu, 0) + 1

gpu_lines = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
).splitlines()

candidates = []
for line in gpu_lines:
    idx, free_mem, util = [part.strip() for part in line.split(",")]
    free_mem = int(free_mem)
    util = int(util)
    reserved_jobs = slot_counts.get(idx, 0)
    effective_free = free_mem - reserved_jobs * job_est
    if util > max_util:
        continue
    if reserved_jobs >= max_jobs:
        continue
    if effective_free < reserve + job_est:
        continue
    candidates.append((util, -effective_free, reserved_jobs, idx))

if not candidates:
    sys.exit(1)

candidates.sort()
print(candidates[0][3])
PY
}

log "queue start poll=${POLL_SECONDS}s util<=${MAX_GPU_UTIL} max_jobs_per_gpu=${MAX_NEW_JOBS_PER_GPU} est_mem=${JOB_EST_MEM_MB}MB reserve=${RESERVE_MEM_MB}MB jobs=${#JOBS[@]}"

while true; do
  launched_any=0
  while true; do
    if ! jobs_remaining; then
      log "all jobs have been dispatched"
      exit 0
    fi

    gpu="$(select_gpu || true)"
    if [ -z "${gpu:-}" ]; then
      break
    fi

    if ! job="$(claim_next_job)"; then
      log "all jobs claimed"
      exit 0
    fi

    IFS='|' read -r tag logfile cmd <<< "$job"
    if [ -f "$(job_done_marker "$tag")" ]; then
      log "skip $tag because done marker exists"
      continue
    fi

    launch_job "$tag" "$logfile" "$gpu" "$cmd"
    launched_any=1
    sleep "$SCHEDULE_INTERVAL_SECONDS"
  done

  if [ "$launched_any" -eq 0 ]; then
    log "no safe gpu slot now; sleep ${POLL_SECONDS}s"
  fi
  sleep "$POLL_SECONDS"
done
