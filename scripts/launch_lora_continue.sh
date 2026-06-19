#!/usr/bin/env bash
# Continuation queue (GPU 3 only):
#   All jobs run sequentially on GPU 3, one at a time.
#   This script is launched after svd_k2 is started externally on GPU 3.
#   It waits for svd_k2 to finish, then runs the remaining 5 jobs in order.
#
#   Queue: svd_k2 (already running) → matched_locked_no_ft → matched_M_a300k
#          → force_locked_no_ft → force_locked_M_a300k → svd_k5
set -euo pipefail
cd /home/nvidia/glm-locking
mkdir -p logs/lora_fair results/hvue_lora_shards

PY=/home/nvidia/miniconda3/envs/evo/bin/python3
SCRIPT=scripts/hvue_lora_finetune.py
TASKS="Host_Tropism Pathogenecity Transmissibility"
GPU=3

run() {
  local ckpt=$1 cfg=$2 logname=$3
  local out="results/hvue_lora_shards/${logname}.csv"
  echo "[launch gpu=$GPU] $ckpt | $cfg -> $out"
  CUDA_VISIBLE_DEVICES=$GPU $PY $SCRIPT \
    --tasks $TASKS \
    --ckpts $ckpt \
    --lrs 1e-4 5e-5 1e-5 \
    --lora_config $cfg \
    --out $out \
    --resume \
    >> logs/lora_fair/${logname}.log 2>&1
  echo "[done] $logname"
}

SVD_K2_PID=$1
echo "Waiting for svd_k2 (PID=$SVD_K2_PID) to finish on GPU $GPU..."
while kill -0 "$SVD_K2_PID" 2>/dev/null; do
  sleep 60
done
echo "svd_k2 done."

run locked_no_ft  matched      matched_locked_no_ft
run M_a300k       matched      matched_M_a300k
run locked_no_ft  force_locked force_locked_no_ft
run M_a300k       force_locked force_locked_M_a300k
run svd_k5_a30k   matched      svd_k5

echo "All jobs on GPU $GPU complete."
