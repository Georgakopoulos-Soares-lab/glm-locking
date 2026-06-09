#!/bin/bash
# Exp 3A — Run ViroBench discriminative evaluation.
#
# Usage:
#   bash experiments/exp3_virobench/run_discriminative.sh [GPU_ID]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPU="${1:-0}"
PYTHON=/home/nvidia/miniconda3/envs/evo/bin/python3
VIROBENCH_DIR="$SCRIPT_DIR/ViroBench"

echo "=== Exp 3A — ViroBench Discriminative Evaluation ==="
echo ""

if [ ! -d "$VIROBENCH_DIR" ]; then
    echo "ERROR: ViroBench not found at $VIROBENCH_DIR"
    echo "Clone it: git clone https://github.com/QIANJINYDX/ViroBench $VIROBENCH_DIR"
    exit 1
fi

if [ ! -d "$VIROBENCH_DIR/data" ]; then
    echo "ERROR: ViroBench data not downloaded."
    echo "Follow instructions in $VIROBENCH_DIR/README.md"
    echo "STOP-AND-ASK: If ViroBench data is access-gated, do not proceed."
    exit 1
fi

echo "ViroBench found. Data directory: $VIROBENCH_DIR/data"
echo ""

# ── Map ViroBench tasks to our probe protocol ────────────────
# For each ViroBench task directory, we:
#   1. Identify the sequence files (train/val/test splits)
#   2. Extract embeddings using hvue_extract_one_ckpt.py
#   3. Train linear probe
#   4. Report per-task metrics

# This is a framework — specific task names depend on ViroBench's actual structure.
echo "ViroBench task directories:"
ls "$VIROBENCH_DIR/data/" 2>/dev/null || echo "  (no data/ subdir — check ViroBench structure)"
echo ""

echo "EVALUATION FRAMEWORK (run per-task):"
echo ""
echo "For each model in [Unlocked-FT, M, Pretrained]:"
echo "  for each ViroBench task:"
echo "    1. Extract embeddings:"
echo "       CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/hvue_extract_one_ckpt.py \\"
echo "         --ckpt_name <model>_<task> \\"
echo "         --ckpt_path <checkpoint_path> \\"
echo "         --data_dir $VIROBENCH_DIR/data/<task>/"
echo "    2. Train probe:"
echo "       $PYTHON scripts/hvue_probe.py \\"
echo "         --emb_dir results/hvue_embeddings \\"
echo "         --out experiments/exp3_virobench/discriminative_results.csv"
echo ""
echo "NOTE: The exact command structure depends on ViroBench's data format."
echo "      Adapt after inspecting the downloaded data."
echo ""
echo "Output: experiments/exp3_virobench/discriminative_results.csv"
