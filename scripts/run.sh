#!/bin/bash
# run.sh — dispatch a lock or finetune job from a config file.
#
# Usage:
#   bash scripts/run.sh lock     configs/lock_v10.yaml
#   bash scripts/run.sh finetune configs/ft_attack_v10.yaml
#
# Multi-GPU is detected automatically (torchrun if >1 GPU).
# Works standalone or sourced by a SLURM script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <lock|finetune> <config.yaml>" >&2
    exit 1
fi

MODE="$1"
CONFIG="$2"

case "$MODE" in
    lock)     PYSCRIPT="scripts/lock.py" ;;
    finetune) PYSCRIPT="scripts/finetune.py" ;;
    *)        echo "[ERROR] Unknown mode '$MODE'. Use 'lock' or 'finetune'." >&2; exit 1 ;;
esac

[[ -f "$CONFIG"   ]] || { echo "[ERROR] Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$PYSCRIPT" ]] || { echo "[ERROR] Script not found: $PYSCRIPT" >&2; exit 1; }

# ---------------------------------------------------------------------------
# GPU count
# ---------------------------------------------------------------------------
NGPUS="${SLURM_GPUS_ON_NODE:-$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)}"

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
