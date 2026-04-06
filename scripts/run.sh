#!/bin/bash
# run.sh — dispatch a lock or finetune job from a config file.
#
# Usage:
#   bash scripts/run.sh lock     configs/lock_v10.yaml            # single GPU (default)
#   bash scripts/run.sh finetune configs/ft_attack_v10.yaml       # single GPU (default)
#   bash scripts/run.sh lock     configs/lock_v10.yaml --multi-gpu  # all available GPUs
#
# Single GPU is the default because SpecDef's SVD term is not data-parallel:
# adding more GPUs adds PCIe allreduce overhead without proportional speedup.
# Use --multi-gpu only on NVLink systems or when the model doesn't fit on 1 GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <lock|finetune> <config.yaml> [--multi-gpu]" >&2
    exit 1
fi

MODE="$1"
CONFIG="$2"
MULTI_GPU=0
if [[ "${3:-}" == "--multi-gpu" ]]; then
    MULTI_GPU=1
fi

case "$MODE" in
    lock)     PYSCRIPT="scripts/lock.py" ;;
    finetune) PYSCRIPT="scripts/finetune.py" ;;
    *)        echo "[ERROR] Unknown mode '$MODE'. Use 'lock' or 'finetune'." >&2; exit 1 ;;
esac

[[ -f "$CONFIG"   ]] || { echo "[ERROR] Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$PYSCRIPT" ]] || { echo "[ERROR] Script not found: $PYSCRIPT" >&2; exit 1; }

# ---------------------------------------------------------------------------
# GPU count — default 1, use all only if --multi-gpu
# ---------------------------------------------------------------------------
if [[ "$MULTI_GPU" -eq 1 ]]; then
    NGPUS="${SLURM_GPUS_ON_NODE:-$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)}"
else
    NGPUS=1
fi

echo "============================================================"
echo " run.sh  mode=${MODE}  config=${CONFIG}"
echo " GPUs: ${NGPUS}  |  $(date)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
if [[ "$NGPUS" -gt 1 ]]; then
    torchrun \
        --standalone \
        --nproc_per_node="${NGPUS}" \
        "$PYSCRIPT" --config "$CONFIG"
else
    python -u "$PYSCRIPT" --config "$CONFIG"
fi
