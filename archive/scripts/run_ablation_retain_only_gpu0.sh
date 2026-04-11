#!/usr/bin/env bash
# GPU 0 — retain-only ablation: lock (alpha=1.0) → finetune 3k steps
# PURPOSE: ablation to separate spectral inflation effect from plain retain-finetuning degradation
# COMPARE with: ft_attack_v11_strong_3k (gap +0.106) to isolate true lock effect

set -e
cd "$(dirname "$0")/.."

LOCK_CFG="configs/lock_ablation_retain_only.yaml"
FT_CFG="configs/ft_attack_ablation_retain_only_3k.yaml"
LOCK_CKPT="results/lock_ablation_retain_only/model_locked.pt"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Step 1: Lock (retain-only, alpha=1.0 throughout)
if [ -f "$LOCK_CKPT" ]; then
    echo "[SKIP] Lock checkpoint already exists: $LOCK_CKPT"
else
    echo "[STEP 1] Running retain-only lock on GPU 0..."
    conda run -n evo --no-capture-output \
        python -u scripts/lock.py --config "$LOCK_CFG" \
        2>&1 | tee "$LOG_DIR/lock_ablation_retain_only_$(date +%Y%m%d_%H%M%S).log"
    echo "[STEP 1] Done."
fi

# Step 2: Finetune attack (locked from retain-only ckpt + unlocked baseline)
echo "[STEP 2] Running finetune attack on GPU 0..."
conda run -n evo --no-capture-output \
    python -u scripts/finetune.py --config "$FT_CFG" \
    2>&1 | tee "$LOG_DIR/ft_ablation_retain_only_3k_$(date +%Y%m%d_%H%M%S).log"
echo "[STEP 2] Done."
echo ""
echo "Results in: results/ft_attack_ablation_retain_only_3k_{locked,unlocked}/metrics.csv"
echo "Compare gap here vs ft_attack_v11_strong_3k gap (+0.106) to isolate spectral effect."
