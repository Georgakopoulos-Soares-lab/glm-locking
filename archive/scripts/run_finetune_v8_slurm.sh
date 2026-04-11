#!/bin/bash
# SLURM launcher — finetune attack on v8 locked + unlocked (comparison).
#
# Runs two sequential stages:
#   Stage 1: finetune on locked   v8 checkpoint (configs/ft_locked_v8_1000lock_topk5_20ep.yaml)
#   Stage 2: finetune on unlocked baseline      (configs/ft_unlocked_v8_1000lock_topk5_20ep.yaml)
#
# Submit:
#   sbatch scripts/run_finetune_v8_slurm.sh

#SBATCH -J ft_v8
#SBATCH -o logs/ft_v8.%j.out
#SBATCH -e logs/ft_v8.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH --mem=40G
#SBATCH -t 2:00:00
#SBATCH -A BCS25073

set -euo pipefail

module purge
unset LD_PRELOAD

CONDA_ROOT="/work/10906/arisk/conda"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate evo
elif [ -f "$CONDA_ROOT/bin/activate" ]; then
    source "$CONDA_ROOT/bin/activate" evo
else
    echo "[ERROR] Cannot find conda at $CONDA_ROOT" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/work/10906/arisk/ls6/evo-locking:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /work/10906/arisk/ls6/evo-locking
mkdir -p logs

# Guard: ensure v8 checkpoint exists before spending queue time
[[ -f "results/lock_v8_all_linear/model_locked.pt" ]] || {
    echo "[ERROR] v8 checkpoint not found: results/lock_v8_all_linear/model_locked.pt" >&2
    exit 1
}

LOG="logs/ft_v8_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " Finetune attack on v8 locked checkpoint (1000 steps, top_k=5)"
echo " Stage 1: locked    — configs/ft_locked_v8_1000lock_topk5_20ep.yaml"
echo " Stage 2: unlocked  — configs/ft_unlocked_v8_1000lock_topk5_20ep_dev.yaml"
echo " $(date)"
echo "============================================================"
echo ""

# echo ">>> [STAGE 1] Finetune LOCKED  ($(date))"
# bash scripts/run.sh finetune configs/ft_locked_v8_1000lock_topk5_20ep.yaml
# echo ""
# echo "    Locked finetune complete."
# echo ""

echo ">>> [STAGE 2] Finetune UNLOCKED  ($(date))"
bash scripts/run.sh finetune configs/ft_unlocked_v8_1000lock_topk5_20ep_dev.yaml
echo ""
echo "    Unlocked finetune complete."
echo ""

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

locked_final,   locked_best   = last_val("results/ft_locked_v8_1000lock_topk5_20ep/metrics.csv")
unlocked_final, unlocked_best = last_val("results/ft_unlocked_v8_1000lock_topk5_20ep_dev/metrics.csv")

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
