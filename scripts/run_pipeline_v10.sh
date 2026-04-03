#!/bin/bash
# SpecDef pipeline v10 — lock → finetune (locked + unlocked).
#
# Supports three GPU tiers:
#   a100  →  8 target blocks  (fits A100-40GB, ~31GB peak)
#   h100  → 24 target blocks (fits H100-80GB, ~65GB peak)
#   full  → 32 target blocks, top_k=5, 10k lock + 20k FT (H100-80GB, ~79GB peak)
#
# Submit examples:
#   sbatch -p gpu-a100     --gres=gpu:1 scripts/run_pipeline_v10.sh a100
#   sbatch -p gpu-a100-dev --gres=gpu:1 scripts/run_pipeline_v10.sh a100
#   sbatch -p h100         --gres=gpu:1 scripts/run_pipeline_v10.sh h100
#   sbatch -p h100         --gres=gpu:1 scripts/run_pipeline_v10.sh full
#
# Skip re-locking if checkpoint already exists:
#   sbatch -p h100 --gres=gpu:1 scripts/run_pipeline_v10.sh h100 --skip-lock

#SBATCH -J evo_v10
#SBATCH -o logs/evo_v10.%j.out
#SBATCH -e logs/evo_v10.%j.err
#SBATCH -N 1
#SBATCH --mem=200G
#SBATCH -t 48:00:00
#SBATCH -A BCS25105

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
GPU_TIER="${1:-}"
shift 2>/dev/null || true  # shift past GPU_TIER

SKIP_LOCK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-lock) SKIP_LOCK=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$GPU_TIER" != "a100" && "$GPU_TIER" != "h100" && "$GPU_TIER" != "full" ]]; then
    echo "Usage: $0 <a100|h100|full> [--skip-lock]"
    echo ""
    echo "  a100 →  8 target blocks (fits A100-40GB)"
    echo "  h100 → 24 target blocks (fits H100-80GB)"
    echo "  full → 32 target blocks, top_k=5, 10k+20k steps (H100-80GB)"
    exit 1
fi

# Select configs based on GPU tier
LOCK_CONFIG="configs/lock_v10_${GPU_TIER}.yaml"
FT_CONFIG="configs/ft_attack_v10_${GPU_TIER}.yaml"
LOCKED_CKPT="results/lock_v10_${GPU_TIER}/model_locked.pt"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs
LOG="logs/pipeline_v10_${GPU_TIER}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " SpecDef Pipeline v10  [${GPU_TIER^^}]"
echo "   lock config: $LOCK_CONFIG"
echo "   ft config:   $FT_CONFIG"
echo "   locked ckpt: $LOCKED_CKPT"
echo "   $(date)  |  Project: $PROJECT_DIR"
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

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ---------------------------------------------------------------------------
# STAGE 1 — Lock
# ---------------------------------------------------------------------------
if [[ "$SKIP_LOCK" -eq 1 ]]; then
    echo ">>> [STAGE 1] Skipping lock (--skip-lock)"
    [[ -f "$LOCKED_CKPT" ]] || { echo "[ERROR] Checkpoint not found: $LOCKED_CKPT" >&2; exit 1; }
    echo "    Using existing checkpoint: $LOCKED_CKPT"
else
    echo ">>> [STAGE 1] Locking model  ($(date))"
    LOCK_START=$SECONDS
    bash scripts/run.sh lock "$LOCK_CONFIG"
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
bash scripts/run.sh finetune "$FT_CONFIG"
FT_ELAPSED=$(( SECONDS - FT_START ))
echo "    Finetune complete in $(( FT_ELAPSED / 60 ))m $(( FT_ELAPSED % 60 ))s"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL_ELAPSED=$(( SECONDS ))
echo "============================================================"
echo " RESULTS SUMMARY  [${GPU_TIER^^}]  ($(date))"
echo " Total wall time: $(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m"
echo "============================================================"

python -u - "$GPU_TIER" <<'PYEOF'
import csv, os, sys

gpu_tier = sys.argv[1]
base = f"ft_attack_v10_{gpu_tier}"

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

echo ""
echo "Full log: $LOG"
echo "Done."
