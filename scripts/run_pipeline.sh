#!/bin/bash

#SBATCH -J evo_lock
#SBATCH -o logs/evo_lock.%j.out
#SBATCH -e logs/evo_lock.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --mem=40G
#SBATCH -t 48:00:00
#SBATCH -A BCS25073

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RUN_NAME="v9"
SKIP_LOCK=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)   RUN_NAME="$2"; shift 2 ;;
        --skip-lock)  SKIP_LOCK=1;   shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

LOCK_RUN="lock_${RUN_NAME}"
FT_LOCKED_RUN="ft_locked_${RUN_NAME}"
FT_UNLOCKED_RUN="ft_unlocked_${RUN_NAME}"

LOCK_DIR="results/${LOCK_RUN}"
FT_LOCKED_DIR="results/${FT_LOCKED_RUN}"
FT_UNLOCKED_DIR="results/${FT_UNLOCKED_RUN}"
LOCKED_CKPT="${LOCK_DIR}/model_locked.pt"

mkdir -p logs
LOG="logs/pipeline_${RUN_NAME}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " SpecDef Pipeline  run=${RUN_NAME}"
echo " $(date)"
echo " Project: $PROJECT_DIR"
echo " Lock dir:        $LOCK_DIR"
echo " FT locked dir:   $FT_LOCKED_DIR"
echo " FT unlocked dir: $FT_UNLOCKED_DIR"
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
    echo "[ERROR] Cannot find conda at $CONDA_ROOT" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

PYTHON="python -u"

# ---------------------------------------------------------------------------
# GPU count
# ---------------------------------------------------------------------------
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo "GPUs available: $NGPUS"
echo ""

# ---------------------------------------------------------------------------
# Helper: run a python script with inline config overrides via temp copy
# ---------------------------------------------------------------------------
run_python_with_config() {
    local script="$1"
    local tmpscript
    tmpscript="$(mktemp /tmp/pipeline_script_XXXXXX.py)"
    trap 'rm -f "$tmpscript"' RETURN
    cp "$script" "$tmpscript"

    # Apply overrides: remaining args are "PATTERN|REPLACEMENT" pairs.
    # Use Python for substitution — sed breaks when pattern contains double quotes.
    shift
    for override in "$@"; do
        local pattern="${override%%|*}"
        local replacement="${override#*|}"
        python3 -c "
import sys
p, r, path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f: c = f.read()
assert p in c, 'Pattern not found in script: ' + repr(p)
with open(path, 'w') as f: f.write(c.replace(p, r))
" "$pattern" "$replacement" "$tmpscript"
    done

    if [[ "$NGPUS" -gt 1 ]]; then
        torchrun --standalone --nproc_per_node="$NGPUS" "$tmpscript"
    else
        $PYTHON "$tmpscript"
    fi
}

# ---------------------------------------------------------------------------
# STAGE 1 — Lock
# ---------------------------------------------------------------------------
if [[ "$SKIP_LOCK" -eq 1 ]]; then
    echo ">>> [STAGE 1] Skipping lock (--skip-lock set)"
    [[ -f "$LOCKED_CKPT" ]] || { echo "[ERROR] Expected checkpoint not found: $LOCKED_CKPT" >&2; exit 1; }
    echo "    Using existing checkpoint: $LOCKED_CKPT"
else
    echo ">>> [STAGE 1] Locking model  ($(date))"
    echo "    run_name=${LOCK_RUN}  results_dir=${LOCK_DIR}"
    echo ""

    run_python_with_config scripts/lock.py \
        "run_name=\"lock_v9_topk4\"|run_name=\"${LOCK_RUN}\"" \
        "results_dir=\"results/lock_v9_topk4\"|results_dir=\"${LOCK_DIR}\""

    [[ -f "$LOCKED_CKPT" ]] || { echo "[ERROR] Lock did not produce checkpoint: $LOCKED_CKPT" >&2; exit 1; }
    echo ""
    echo "    Lock complete. Checkpoint: $LOCKED_CKPT"
fi

echo ""

# ---------------------------------------------------------------------------
# STAGE 2 — Finetune attack on LOCKED model
# ---------------------------------------------------------------------------
echo ">>> [STAGE 2] Finetune attack — LOCKED  ($(date))"
echo "    run_name=${FT_LOCKED_RUN}  results_dir=${FT_LOCKED_DIR}"
echo "    locked_ckpt=${LOCKED_CKPT}"
echo ""

run_python_with_config scripts/finetune.py \
    "run_name=\"ft_attack_locked_v9\"|run_name=\"${FT_LOCKED_RUN}\"" \
    "results_dir=\"results/ft_attack_locked_v9\"|results_dir=\"${FT_LOCKED_DIR}\"" \
    "locked_ckpt=\"results/lock_v9_topk4/model_locked.pt\"|locked_ckpt=\"${LOCKED_CKPT}\""

echo ""
echo "    Locked finetune complete."
echo ""

# ---------------------------------------------------------------------------
# STAGE 3 — Finetune attack on UNLOCKED (baseline) model
# ---------------------------------------------------------------------------
echo ">>> [STAGE 3] Finetune attack — UNLOCKED baseline  ($(date))"
echo "    run_name=${FT_UNLOCKED_RUN}  results_dir=${FT_UNLOCKED_DIR}"
echo ""

run_python_with_config scripts/finetune.py \
    "run_name=\"ft_attack_locked_v9\"|run_name=\"${FT_UNLOCKED_RUN}\"" \
    "results_dir=\"results/ft_attack_locked_v9\"|results_dir=\"${FT_UNLOCKED_DIR}\"" \
    "locked_ckpt=\"results/lock_v9_topk4/model_locked.pt\"|locked_ckpt=None"

echo ""
echo "    Unlocked finetune complete."
echo ""

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------
echo "============================================================"
echo " RESULTS SUMMARY  ($(date))"
echo "============================================================"
echo ""

compare_csv() {
    local label="$1"
    local csv="$2"
    if [[ -f "$csv" ]]; then
        echo "  [$label]  $csv"
        # Print header + last 5 rows
        head -1 "$csv"
        tail -5 "$csv"
    else
        echo "  [$label]  NOT FOUND: $csv"
    fi
    echo ""
}

compare_csv "LOCKED   metrics" "${FT_LOCKED_DIR}/metrics.csv"
compare_csv "UNLOCKED metrics" "${FT_UNLOCKED_DIR}/metrics.csv"

# Quick val_loss comparison at final step
python -u - <<PYEOF
import csv, os

def last_val(path):
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    last = rows[-1]
    best = min(rows, key=lambda r: float(r["val_loss"]))
    return float(last["val_loss"]), float(best["val_loss"])

locked_final, locked_best     = last_val("${FT_LOCKED_DIR}/metrics.csv")
unlocked_final, unlocked_best = last_val("${FT_UNLOCKED_DIR}/metrics.csv")

print("  Final val_loss  —  locked: {:.4f}   unlocked: {:.4f}   gap: {:.4f}".format(
    locked_final or 0, unlocked_final or 0,
    (locked_final or 0) - (unlocked_final or 0)
))
print("  Best  val_loss  —  locked: {:.4f}   unlocked: {:.4f}   gap: {:.4f}".format(
    locked_best or 0, unlocked_best or 0,
    (locked_best or 0) - (unlocked_best or 0)
))
PYEOF

echo ""
echo "Full log: $LOG"
echo "Done."
