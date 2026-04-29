#!/bin/bash
# Run a complete per-checkpoint pipeline serially on a single GPU:
#   1) Fine-tune from a config
#   2) Compute held-out attack PPL on the resulting ckpt
#   3) Extract HVUE embeddings (3 tasks x 2 splits) for downstream probe
#
# All three steps share one GPU and run sequentially in the same tmux job.
# Logs go to logs/runs/<run_name>.log
#
# Usage:  scripts/run_pipeline.sh <gpu_id> <config_yaml>

set -e
set -o pipefail

GPU=${1:?gpu id}
CFG=${2:?finetune config yaml}
PY=/home/nvidia/miniconda3/envs/evo/bin/python

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
echo "[$(date)] PIPELINE START   GPU=$GPU  cfg=$CFG  run=$ACTUAL_RUN_NAME" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

# ---- Step 1: fine-tune ---------------------------------------------------
echo "[$(date)] >>> Step 1: fine-tune" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/finetune.py --config "$CFG" 2>&1 | tee -a "$LOG"

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
