#!/bin/bash
# SpecDef pipeline v10 — alpha=0.99 variant (paper value, fixed retain weight).
# Same as run_pipeline_v10.sh but with alpha_start=alpha_end=0.99 per the paper's
# text-classification experiments (Appendix B.2, locking.pdf).
# Pinned to GPU 7 of the current node.
#
# Usage:
#   bash scripts/run_pipeline_v10_alpha099_gpu7.sh
#   bash scripts/run_pipeline_v10_alpha099_gpu7.sh --skip-lock

set -euo pipefail

# ---------------------------------------------------------------------------
# Pin to GPU 7
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=7

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
LOCK_CONFIG="configs/lock_v10_full_alpha099.yaml"
FT_CONFIG="configs/ft_attack_v10_full_alpha099.yaml"
LOCKED_CKPT="results/lock_v10_full_alpha099/model_locked.pt"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
SKIP_LOCK=0
MULTI_GPU_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-lock)  SKIP_LOCK=1; shift ;;
        --multi-gpu)  MULTI_GPU_FLAG="--multi-gpu"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs
LOG="logs/pipeline_v10_alpha099_gpu7_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " SpecDef Pipeline v10 — alpha=0.99 (paper value)  [GPU 7]"
echo "   lock config: $LOCK_CONFIG"
echo "   ft config:   $FT_CONFIG"
echo "   locked ckpt: $LOCKED_CKPT"
echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "   $(date)  |  Project: $PROJECT_DIR"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Conda env
# ---------------------------------------------------------------------------
CONDA_ROOT="/home/nvidia/miniconda3"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate evo
elif [ -f "$CONDA_ROOT/bin/activate" ]; then
    source "$CONDA_ROOT/bin/activate" evo
else
    echo "[ERROR] Could not find conda initialization at $CONDA_ROOT. Edit CONDA_ROOT in this script." >&2
    exit 1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ---------------------------------------------------------------------------
# STAGE 1 — Lock
# ---------------------------------------------------------------------------
if [[ "$SKIP_LOCK" -eq 0 ]] && [[ -f "$LOCKED_CKPT" ]]; then
    echo ">>> [STAGE 1] Locked checkpoint already exists — skipping lock"
    echo "    Found: $LOCKED_CKPT"
    SKIP_LOCK=1
fi

if [[ "$SKIP_LOCK" -eq 1 ]]; then
    echo ">>> [STAGE 1] Skipping lock (--skip-lock)"
    [[ -f "$LOCKED_CKPT" ]] || { echo "[ERROR] Checkpoint not found: $LOCKED_CKPT" >&2; exit 1; }
    echo "    Using existing checkpoint: $LOCKED_CKPT"
else
    echo ">>> [STAGE 1] Locking model  ($(date))"
    LOCK_START=$SECONDS
    bash scripts/run.sh lock "$LOCK_CONFIG" $MULTI_GPU_FLAG
    [[ -f "$LOCKED_CKPT" ]] || { echo "[ERROR] Lock did not produce: $LOCKED_CKPT" >&2; exit 1; }
    LOCK_ELAPSED=$(( SECONDS - LOCK_START ))
    echo "    Lock complete in $(( LOCK_ELAPSED / 60 ))m $(( LOCK_ELAPSED % 60 ))s"
    echo "    Checkpoint: $LOCKED_CKPT"
fi
echo ""

# ---------------------------------------------------------------------------
# STAGE 2 — Finetune (mode=both → locked then unlocked)
# ---------------------------------------------------------------------------
echo ">>> [STAGE 2] Finetune LOCKED + UNLOCKED  ($(date))"
FT_START=$SECONDS
bash scripts/run.sh finetune "$FT_CONFIG" $MULTI_GPU_FLAG
FT_ELAPSED=$(( SECONDS - FT_START ))
echo "    Finetune complete in $(( FT_ELAPSED / 60 ))m $(( FT_ELAPSED % 60 ))s"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL_ELAPSED=$(( SECONDS ))
echo "============================================================"
echo " RESULTS SUMMARY  ($(date))"
echo " Total wall time: $(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m"
echo "============================================================"

python -u - <<'PYEOF'
import csv, os

base = "ft_attack_v10_full_alpha099"

def read_metrics(path):
    if not os.path.exists(path): return None, None
    with open(path) as f: rows = list(csv.DictReader(f))
    if not rows: return None, None
    last = rows[-1]
    best = min(rows, key=lambda r: float(r["val_loss"]))
    return float(last["val_loss"]), float(best["val_loss"])

locked_dir   = f"results/{base}_locked/metrics.csv"
unlocked_dir = f"results/{base}_unlocked/metrics.csv"

locked_final,   locked_best   = read_metrics(locked_dir)
unlocked_final, unlocked_best = read_metrics(unlocked_dir)

print(f"\n  {'':30s} {'LOCKED':>10s}  {'UNLOCKED':>10s}  {'GAP':>10s}")
print(f"  {'-'*62}")

if locked_final is not None and unlocked_final is not None:
    print(f"  {'Final val_loss':30s} {locked_final:10.4f}  {unlocked_final:10.4f}  {locked_final - unlocked_final:+10.4f}")
    print(f"  {'Best  val_loss':30s} {locked_best:10.4f}  {unlocked_best:10.4f}  {locked_best - unlocked_best:+10.4f}")
    if locked_final > unlocked_final:
        print("\n  SpecDef WORKING: locked model resists attack (higher loss = less memorization)")
    else:
        print("\n  SpecDef NOT working: locked model learned attack equally well")
else:
    print("  [WARNING] Could not read one or both metrics files")
    if locked_final  is not None: print(f"  Locked:   final={locked_final:.4f}  best={locked_best:.4f}")
    if unlocked_final is not None: print(f"  Unlocked: final={unlocked_final:.4f}  best={unlocked_best:.4f}")
PYEOF
