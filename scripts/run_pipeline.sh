#!/bin/bash
# Run a complete pipeline: fine-tune → PPL eval → HVUE embeddings → probe.
# Intermediate checkpoints (checkpoint_every in config) are also evaluated.
#
# Usage (single GPU):   scripts/run_pipeline.sh <gpu_id> <config_yaml>
# Usage (multi-GPU):    scripts/run_pipeline.sh <gpu_id> <config_yaml> <nproc>
#
#   gpu_id  – GPU to use for single-GPU steps (PPL, extraction, probe); also
#              used as the first GPU when running multi-GPU fine-tuning.
#   nproc   – number of GPUs for torchrun fine-tuning (default: 1 = no torchrun)
#
# Example (8 GPUs, all GPUs visible):
#   scripts/run_pipeline.sh 0 configs/finetune/bypass_a10k_full_unfreeze_25k.yaml 8
#
# Logs go to logs/runs/<config_name>.log

set -e
set -o pipefail

GPU=${1:?gpu id}
CFG=${2:?finetune config yaml}
NPROC=${3:-1}   # number of GPUs for fine-tuning (1 = single-process, >1 = torchrun)

# Auto-detect evo conda env python. Works on any cluster as long as
# the 'evo' conda env exists and conda is in PATH.
if command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q '^evo '; then
  PY="conda run -n evo --no-capture-output python"
  TORCHRUN="conda run -n evo --no-capture-output torchrun"
else
  # Fallback: try common hardcoded paths (original cluster / new cluster)
  for candidate in \
    /home/nvidia/miniconda3/envs/evo/bin \
    /work/10906/arisk/conda/envs/evo/bin \
    "$HOME/miniconda3/envs/evo/bin" \
    "$HOME/conda/envs/evo/bin"; do
    if [ -x "$candidate/python" ]; then
      PY="$candidate/python"
      TORCHRUN="$candidate/torchrun"
      break
    fi
  done
  if [ -z "$PY" ]; then
    echo "ERROR: evo conda env not found. Run: bash setup_evo_env.sh" >&2
    exit 1
  fi
fi

cd "$(dirname "$0")/.."

NAME=$(basename "$CFG" .yaml)
RUN_NAME=$(grep -E '^run_name:' "$CFG" | awk '{print $2}')
MODE=$(grep -E '^mode:' "$CFG" | awk '{print $2}')
LOG="logs/runs/${NAME}.log"

# finetune.py auto-appends _locked/_unlocked to run_name and results_dir;
# mirror that here so CKPT and ACTUAL_RUN_NAME point to the real paths.
if   [[ "$MODE" == "locked"   ]]; then ACTUAL_RUN_NAME="${RUN_NAME}_locked"
elif [[ "$MODE" == "unlocked" ]]; then ACTUAL_RUN_NAME="${RUN_NAME}_unlocked"
else                                   ACTUAL_RUN_NAME="$RUN_NAME"
fi
CKPT="results/${ACTUAL_RUN_NAME}/model_best.pt"
mkdir -p logs/runs results

echo "==================================================" | tee -a "$LOG"
echo "[$(date)] PIPELINE START   GPU=$GPU  nproc=$NPROC  cfg=$CFG  run=$ACTUAL_RUN_NAME" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

# ---- Step 1: fine-tune ---------------------------------------------------
echo "[$(date)] >>> Step 1: fine-tune (nproc=$NPROC)" | tee -a "$LOG"
if [[ "$NPROC" -gt 1 ]]; then
  # Multi-GPU: use torchrun; all GPUs must be visible (no CUDA_VISIBLE_DEVICES restriction)
  $TORCHRUN --nproc_per_node="$NPROC" --master_port=29504 \
    scripts/finetune.py --config "$CFG" 2>&1 | tee -a "$LOG"
else
  CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/finetune.py --config "$CFG" 2>&1 | tee -a "$LOG"
fi

if [ ! -f "$CKPT" ]; then
  echo "[$(date)] ERROR: expected ckpt not found at $CKPT" | tee -a "$LOG"
  exit 1
fi

# ---- Step 2: attack heldout PPL ------------------------------------------
echo "[$(date)] >>> Step 2: attack heldout PPL on $CKPT" | tee -a "$LOG"
# The attack_ppl.py script is registry-based; pass a 1-row registry CSV so
# we evaluate exactly this new ckpt without disturbing the rest of the table.
TMP=$(mktemp --suffix=.csv)
echo "name,ckpt" > "$TMP"
echo "${ACTUAL_RUN_NAME},${CKPT}" >> "$TMP"
# Serialize CSV writes across concurrent pipelines via flock on a sibling lockfile.
LOCK="results/attack_heldout_ppl.csv.lock"
touch "$LOCK"
( flock -x 200
  CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/attack_ppl.py \
    --n_batches 64 \
    --ckpts_csv "$TMP" \
    --out "results/attack_heldout_ppl.csv" 2>&1 | tee -a "$LOG"
) 200>"$LOCK"
rm -f "$TMP"

# ---- Step 3: HVUE embedding extraction -----------------------------------
# hvue_extract_one_ckpt.py iterates the 3 tasks x 2 splits internally.
echo "[$(date)] >>> Step 3: HVUE embedding extraction" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/hvue_extract_one_ckpt.py \
  --ckpt_name "$ACTUAL_RUN_NAME" \
  --ckpt_path "$CKPT" 2>&1 | tee -a "$LOG"

# ---- Step 4: HVUE linear probe (on all embeddings accumulated so far) -----
echo "[$(date)] >>> Step 4: HVUE linear probe" | tee -a "$LOG"
LOCK_PROBE="results/hvue_probe.csv.lock"
touch "$LOCK_PROBE"
( flock -x 201
  $PY -u scripts/hvue_probe.py \
    --emb_dir results/hvue_embeddings \
    --out results/hvue_probe.csv 2>&1 | tee -a "$LOG"
) 201>"$LOCK_PROBE"

# ---- Step 5: HVUE pipeline on intermediate checkpoints (if any) ----------
CKPT_DIR="results/${ACTUAL_RUN_NAME}/checkpoints"
if [[ -d "$CKPT_DIR" ]]; then
  INTER_CKPTS=( $(ls "$CKPT_DIR"/*.pt 2>/dev/null | sort) )
  if [[ ${#INTER_CKPTS[@]} -gt 0 ]]; then
    echo "[$(date)] >>> Step 5: intermediate checkpoint evaluation (${#INTER_CKPTS[@]} ckpts)" | tee -a "$LOG"
    for INTER in "${INTER_CKPTS[@]}"; do
      INTER_NAME="${ACTUAL_RUN_NAME}__$(basename "$INTER" .pt)"
      echo "[$(date)]   evaluating $INTER_NAME" | tee -a "$LOG"
      CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/hvue_extract_one_ckpt.py \
        --ckpt_name "$INTER_NAME" \
        --ckpt_path "$INTER" 2>&1 | tee -a "$LOG"
    done
    # Re-run probe to include intermediate ckpts in summary table
    ( flock -x 201
      $PY -u scripts/hvue_probe.py \
        --emb_dir results/hvue_embeddings \
        --out results/hvue_probe.csv 2>&1 | tee -a "$LOG"
    ) 201>"$LOCK_PROBE"
  fi
fi

echo "[$(date)] PIPELINE DONE  $ACTUAL_RUN_NAME" | tee -a "$LOG"
