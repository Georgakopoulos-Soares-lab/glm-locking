#!/usr/bin/env bash
# Launch the fair LoRA head-to-head + SVD runs.
# Each (ckpt × config) writes to its own shard CSV (no concurrent-write races);
# merge shards at the end with a separate script.
#
# 9 jobs total, 7 GPUs (1-7) available (GPU 0 in use by another workload).
# Wave 1: 7 jobs in parallel. Wave 2: remaining 2 SVD jobs.
set -euo pipefail
cd /home/nvidia/glm-locking
mkdir -p logs/lora_fair results/hvue_lora_shards

PY=/home/nvidia/miniconda3/envs/evo/bin/python3
SCRIPT=scripts/hvue_lora_finetune.py
TASKS="Host_Tropism Pathogenecity Transmissibility"

run() {
  local gpu=$1 ckpt=$2 cfg=$3 logname=$4
  local out="results/hvue_lora_shards/${logname}.csv"
  CUDA_VISIBLE_DEVICES=$gpu $PY $SCRIPT \
    --tasks $TASKS \
    --ckpts $ckpt \
    --lrs 1e-4 5e-5 1e-5 \
    --lora_config $cfg \
    --out $out \
    --resume \
    > logs/lora_fair/${logname}.log 2>&1 &
  echo "[launch gpu=$gpu] $ckpt | $cfg → $out  (PID=$!)"
}

# Wave 1: 7 jobs on GPUs 1-7
run 1 pretrained    matched      matched_pretrained
run 2 locked_no_ft  matched      matched_locked_no_ft
run 3 unlocked_ft   matched      matched_unlocked_ft
run 4 M_a300k       matched      matched_M_a300k
run 5 locked_no_ft  force_locked force_locked_no_ft
run 6 M_a300k       force_locked force_locked_M_a300k
run 7 svd_k3_a30k   matched      svd_k3

echo
echo "Wave 1 launched (7 jobs). Waiting for ALL to finish before wave 2..."
wait
echo
echo "Wave 1 complete. Launching wave 2 (2 jobs)..."

# Wave 2: remaining 2 SVD ckpts
run 1 svd_k2_a30k   matched      svd_k2
run 2 svd_k5_a30k   matched      svd_k5

wait
echo "All jobs complete."
