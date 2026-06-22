#!/usr/bin/env bash
# Multiseed LoRA fine-tuning (matched config, 96 layers, LR=1e-4 only).
# 3 checkpoints × 2 extra seeds = 6 jobs.
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
LR=1e-4

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
    echo "[launch gpu=$gpu] $ckpt seed=$seed lr=$LR → $out"
    CUDA_VISIBLE_DEVICES=$gpu $PY $SCRIPT \
        --tasks $TASKS \
        --ckpts $ckpt \
        --lrs $LR \
        --lora_config $CONFIG \
        --seed $seed \
        --out "$out" \
        > "$log" 2>&1 &
    echo "  PID=$!"
}

echo "Launching multiseed experiments (matched config, LR=$LR only)"
echo "=============================================================="
echo "  Checkpoints: ${CKPTS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Total: 3 ckpts × 2 seeds × 3 tasks × 1 LR = 18 runs"
echo ""

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
