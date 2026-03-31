#!/bin/bash
# SpecDef pipeline v2 — lock (top_k=5, 5000 steps) then 2×finetune (20 epochs).
#
# All hyperparameters live in configs/:
#   Stage 1 lock:      configs/lock_topk5_5000steps.yaml
#   Stage 2 ft locked: configs/ft_locked_topk5_5000lock_20ep.yaml
#   Stage 3 ft unlkd:  configs/ft_unlocked_topk5_5000lock_20ep.yaml
#
# Submit:
#   sbatch scripts/run_pipeline_v2.sh
# Skip re-locking if checkpoint already exists:
#   sbatch scripts/run_pipeline_v2.sh --skip-lock

#SBATCH -J evo_lock_v2
#SBATCH -o logs/evo_lock_v2.%j.out
#SBATCH -e logs/evo_lock_v2.%j.err
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH --mem=15G
#SBATCH -t 48:00:00
#SBATCH -A BCS25073

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
SKIP_LOCK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-lock) SKIP_LOCK=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

LOCKED_CKPT="results/lock_topk5_5000steps/model_locked.pt"

mkdir -p logs
LOG="logs/pipeline_v2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " SpecDef Pipeline v2"
echo "   lock:    configs/lock_topk5_5000steps.yaml"
 echo "   ft lock: configs/ft_locked_topk5_5000lock_20ep.yaml"
 echo "   ft base: configs/ft_unlocked_topk5_5000lock_20ep.yaml"
echo " $(date)  |  Project: $PROJECT_DIR"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Conda env
# ---------------------------------------------------------------------------
CONDA_ROOT="${CONDA_ROOT:-/work/10906/arisk/conda}"
if [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate evo
elif [[ -f "$CONDA_ROOT/bin/activate" ]]; then
    source "$CONDA_ROOT/bin/activate" evo
else
    echo "[ERROR] Cannot find conda at $CONDA_ROOT" >&2; exit 1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# STAGE 1 — Lock
# ---------------------------------------------------------------------------
if [[ "$SKIP_LOCK" -eq 1 ]]; then
    echo ">>> [STAGE 1] Skipping lock (--skip-lock)"
    [[ -f "$LOCKED_CKPT" ]] || { echo "[ERROR] Checkpoint not found: $LOCKED_CKPT" >&2; exit 1; }
    echo "    Using existing checkpoint: $LOCKED_CKPT"
else
    echo ">>> [STAGE 1] Locking model  ($(date))"
    bash scripts/run.sh lock configs/lock_topk5_5000steps.yaml
    [[ -f "$LOCKED_CKPT" ]] || { echo "[ERROR] Lock did not produce: $LOCKED_CKPT" >&2; exit 1; }
    echo "    Lock complete. Checkpoint: $LOCKED_CKPT"
fi
echo ""

# ---------------------------------------------------------------------------
# STAGE 2 — Finetune LOCKED
# ---------------------------------------------------------------------------
echo ">>> [STAGE 2] Finetune LOCKED  ($(date))"
bash scripts/run.sh finetune configs/ft_locked_topk5_5000lock_20ep.yaml
echo "    Locked finetune complete."
echo ""

# ---------------------------------------------------------------------------
# STAGE 3 — Finetune UNLOCKED baseline
# ---------------------------------------------------------------------------
echo ">>> [STAGE 3] Finetune UNLOCKED  ($(date))"
bash scripts/run.sh finetune configs/ft_unlocked_topk5_5000lock_20ep.yaml
echo "    Unlocked finetune complete."
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo " RESULTS SUMMARY  ($(date))"
echo "============================================================"

python -u - <<'PYEOF'
import csv, os

def last_val(path):
    if not os.path.exists(path): return None, None
    with open(path) as f: rows = list(csv.DictReader(f))
    if not rows: return None, None
    last = rows[-1]
    best = min(rows, key=lambda r: float(r["val_loss"]))
    return float(last["val_loss"]), float(best["val_loss"])

locked_final,   locked_best   = last_val("results/ft_locked_topk5_5000lock_20ep/metrics.csv")
unlocked_final, unlocked_best = last_val("results/ft_unlocked_topk5_5000lock_20ep/metrics.csv")

print("  Final val_loss  —  locked: {:.4f}   unlocked: {:.4f}   gap: {:.4f}".format(
    locked_final or 0, unlocked_final or 0,
    (locked_final or 0) - (unlocked_final or 0)))
print("  Best  val_loss  —  locked: {:.4f}   unlocked: {:.4f}   gap: {:.4f}".format(
    locked_best or 0, unlocked_best or 0,
    (locked_best or 0) - (unlocked_best or 0)))
PYEOF

echo ""
echo "Full log: $LOG"
echo "Done."
