#!/usr/bin/env bash
# GPU 3 — Direct SVD scaling ×100 lock → finetune 3k steps
# PURPOSE: test if directly scaling σ by ×100 creates a stronger barrier than
#   gradient-based v11_strong lock (which achieved σ_max≈117 after 5000 steps)
# COMPARE: ft_attack_v11_strong_3k gap (+0.106)

set -e
cd "$(dirname "$0")/.."

LOCK_CFG="configs/lock_direct_scale_x100.yaml"
FT_CFG="configs/ft_attack_direct_scale_x100_3k.yaml"
LOCK_CKPT="results/lock_direct_scale_x100/model_locked.pt"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Step 1: Direct SVD scaling + retain recovery
if [ -f "$LOCK_CKPT" ]; then
    echo "[SKIP] Lock checkpoint already exists: $LOCK_CKPT"
else
    echo "[STEP 1] Running direct SVD scaling (x100) + 500-step retain recovery on GPU 3..."
    conda run -n evo --no-capture-output \
        python -u scripts/lock_direct_scale.py --config "$LOCK_CFG" \
        2>&1 | tee "$LOG_DIR/lock_direct_scale_x100_$(date +%Y%m%d_%H%M%S).log"
    echo "[STEP 1] Done."
fi

# Step 2: Finetune attack (locked + unlocked)
echo "[STEP 2] Running finetune attack on GPU 3..."
conda run -n evo --no-capture-output \
    python -u scripts/finetune.py --config "$FT_CFG" \
    2>&1 | tee "$LOG_DIR/ft_direct_scale_x100_3k_$(date +%Y%m%d_%H%M%S).log"
echo "[STEP 2] Done."
echo ""
echo "Results in: results/ft_attack_direct_scale_x100_3k_{locked,unlocked}/metrics.csv"
echo "Compare vs v11_strong_3k gap (+0.106) to test σ_max hypothesis."
