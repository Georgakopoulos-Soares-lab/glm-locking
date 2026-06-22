#!/usr/bin/env bash
# Multiseed LoRA fine-tuning (matched config, 96 layers, fair comparison).
# 3 checkpoints × 2 extra seeds = 6 jobs total.
#
# Seed 0 already completed → these are seeds 1 and 2.
# Each (ckpt, seed) writes to its own shard CSV.
#
# Usage: bash scripts/launch_multiseed.sh
set -euo pipefail
cd /home/nvidia/glm-locking
mkdir -p logs/lora_fair results/hvue_lora_shards

PY=/home/nvidia/miniconda3/envs/evo/bin/python3
SCRIPT=scripts/hvue_lora_finetune.py
TASKS="Host_Tropism Pathogenecity Transmissibility"
CONFIG=matched

# Checkpoints for multiseed
CKPTS=(
    "pretrained"
    "unlocked_ft"
    "svd_k2_a30k"
)
SEEDS=(1 2)

run() {
    local gpu=$1 ckpt=$2 seed=$3
    local out="results/hvue_lora_shards/${ckpt}_s${seed}.csv"
    local log="logs/lora_fair/${ckpt}_s${seed}.log"
    echo "[launch gpu=$gpu] $ckpt seed=$seed → $out"
    CUDA_VISIBLE_DEVICES=$gpu $PY $SCRIPT \
        --tasks $TASKS \
        --ckpts $ckpt \
        --lrs 1e-4 5e-5 1e-5 \
        --lora_config $CONFIG \
        --seed $seed \
        --out "$out" \
        > "$log" 2>&1 &
    echo "  PID=$!"
}

echo "Launching multiseed experiments (matched config, 6 jobs)"
echo "=========================================================="

# Assign GPUs — use whatever is free. Default: GPUs 1-6
GPU=1
for ckpt in "${CKPTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run $GPU "$ckpt" "$seed"
        GPU=$((GPU + 1))
        sleep 3  # stagger model loading to avoid OOM races
    done
done

echo ""
echo "All 6 jobs launched. Waiting..."
wait
echo "All multiseed jobs complete."
