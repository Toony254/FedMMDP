#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/bd/data/zs/FedMMDP-base"
PYTHON_BIN="/mnt/data/software/anaconda/bin/conda"
DATA_ROOT="/home/bd/data/zs/FedMMDP/preprocessed_iapr_siglip/domain_datasets"

cd "$PROJECT_ROOT"

is_running() {
  local name="$1"
  local proj="$2"
  local matches
  matches="$(pgrep -af -- "--name ${name}" || true)"
  if [[ -z "$matches" ]]; then
    return 1
  fi
  while IFS= read -r line; do
    [[ "$line" == *"--dataset iapr"* ]] || continue
    [[ "$line" == *"--model siglip"* ]] || continue
    [[ "$line" == *"--use_pretrained_proj ${proj}"* ]] || continue
    return 0
  done <<< "$matches"
  return 1
}

launch_job() {
  local gpu="$1"
  local proj="$2"
  local proj_tag="$3"
  local name="$4"
  local log_file="$5"
  shift 5

  if is_running "$name" "$proj"; then
    echo "SKIP running: ${name} proj=${proj}"
    return 0
  fi

  local cmd=(
    "$PYTHON_BIN" run -n zs_vita python src/main.py
    --name "$name"
    --dataset iapr
    --data_root "$DATA_ROOT"
    --feature_dim 768
    --lr 1e-5
    --local_epochs 1
    --comm_rounds 50
    --model siglip
    --batch_size 256
    --use_pretrained_proj "$proj"
    --partition hetero
    "$@"
  )

  local quoted_cmd
  printf -v quoted_cmd '%q ' "${cmd[@]}"
  nohup bash -lc "cd '$PROJECT_ROOT' && CUDA_VISIBLE_DEVICES=${gpu} ${quoted_cmd} > 'outputs/${log_file}' 2>&1" >/dev/null 2>&1 &
  echo "LAUNCHED gpu=${gpu} proj=${proj_tag} name=${name} log=outputs/${log_file}"
  sleep 1
}

launch_set() {
  local proj="$1"
  local proj_tag="$2"

  launch_job 0 "$proj" "$proj_tag" "FedMMDP-avg-siglip"      "output_avg_siglip_iapr_${proj_tag}.log"      --FL_algorithm FedAvg
  launch_job 0 "$proj" "$proj_tag" "FedMMDP-prox-siglip"     "output_prox_siglip_iapr_${proj_tag}.log"     --FL_algorithm FedProx
  launch_job 0 "$proj" "$proj_tag" "FedMMDP-md-siglip"       "output_md_siglip_iapr_${proj_tag}.log"      --FL_algorithm FedMD --pub_data_num 5000

  launch_job 1 "$proj" "$proj_tag" "FedMMDP-df-siglip"       "output_df_siglip_iapr_${proj_tag}.log"      --FL_algorithm FedDF --pub_data_num 5000
  launch_job 1 "$proj" "$proj_tag" "FedMMDP-Cream-siglip"    "output_cream_siglip_iapr_${proj_tag}.log"   --FL_algorithm Cream --pub_data_num 5000
  launch_job 1 "$proj" "$proj_tag" "FedMMDP-Harmony-siglip"  "output_Harmony_siglip_iapr_${proj_tag}.log" --FL_algorithm Harmony --harmony_stage1_rounds 10 --harmony_clusters 2 --harmony_cluster_mode fixed

  launch_job 2 "$proj" "$proj_tag" "FedMMDP-MASA-siglip"     "output_MASA_siglip_iapr_${proj_tag}.log"    --FL_algorithm MASA
  launch_job 2 "$proj" "$proj_tag" "FedMMDP-MEMA-siglip"     "output_FedMEMA_siglip_iapr_${proj_tag}.log" --FL_algorithm FedMEMA

  launch_job 3 "$proj" "$proj_tag" "FedMMDP-Mobile-siglip"   "output_FedMobile_siglip_iapr_${proj_tag}.log" --FL_algorithm FedMobile
  launch_job 3 "$proj" "$proj_tag" "FedMMDP-MEKT-siglip"     "output_FedMEKT_siglip_iapr_${proj_tag}.log" --FL_algorithm FedMEKT
  launch_job 3 "$proj" "$proj_tag" "FedMMDP-siglip"          "output_FedMMDP_siglip_iapr_${proj_tag}.log" --FL_algorithm FedMMDP
}

launch_set 1 "preproj"
launch_set 0 "randproj"

echo "DONE"
