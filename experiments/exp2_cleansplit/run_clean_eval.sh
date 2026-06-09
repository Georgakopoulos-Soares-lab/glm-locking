#!/bin/bash
# Exp 2 — Clean PPL evaluation on the final-test set for all paper checkpoints.
#
# Prerequisites:
#   python3 experiments/split_manifest/build_three_way_split.py
#
# Usage:
#   bash experiments/exp2_cleansplit/run_clean_eval.sh [GPU_ID]
#
# Output:
#   experiments/exp2_cleansplit/clean_ppl_results.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPU="${1:-0}"
PYTHON="${REPO_ROOT}/miniconda3/envs/evo/bin/python3"
# If the above path doesn't exist, fall back to system python3
if [ ! -f "$PYTHON" ]; then
    PYTHON=/home/nvidia/miniconda3/envs/evo/bin/python3
fi

TEST_FASTA="$REPO_ROOT/experiments/split_manifest/final_test.fasta"
if [ ! -f "$TEST_FASTA" ]; then
    echo "ERROR: $TEST_FASTA not found. Run build_three_way_split.py first."
    exit 1
fi

echo "=== Exp 2 Clean PPL Evaluation ==="
echo "Test set: $TEST_FASTA"
echo "GPU: $GPU"
echo ""

# ── Checkpoint list (label, ckpt_name, ckpt_path_or_pretrained) ──
declare -A CKPTS
CKPTS=(
    ["Pretrained"]="pretrained|pretrained"
    ["Unlocked"]="ft_unlocked_25k_v2_unlocked|results/ft_unlocked_25k_v2_unlocked/model_best.pt"
    ["A_a10k_lr1e6"]="ft_locked_a10k_lr1e6_25k_locked|results/ft_locked_a10k_lr1e6_25k_locked/model_best.pt"
    ["B_a30k_lr1e6"]="ft_locked_a30k_lr1e6_25k_locked|results/ft_locked_a30k_lr1e6_25k_locked/model_best.pt"
    ["Cprime_a30k_lr1e5"]="ft_locked_a30k_lr1e5_25k_locked|results/ft_locked_a30k_lr1e5_25k_locked/model_best.pt"
    ["D_lora"]="ft_lora_a10k_25k_locked|results/ft_lora_a10k_25k_locked/model_best.pt"
    ["E_bypass_a10k"]="ft_bypass_a10k_25k_v2_locked|results/ft_bypass_a10k_25k_v2_locked/model_best.pt"
    ["F_bypass_a30k"]="ft_bypass_a30k_25k_locked|results/ft_bypass_a30k_25k_locked/model_best.pt"
    ["G_svdk3_a10k"]="ft_theorem8_a10k_k3_25k_locked|results/ft_theorem8_a10k_k3_25k_locked/model_best.pt"
    ["H_svdk2_a30k"]="ft_theorem8_a30k_k2_25k_locked|results/ft_theorem8_a30k_k2_25k_locked/model_best.pt"
    ["I_svdk3_a30k"]="ft_theorem8_a30k_k3_25k_locked|results/ft_theorem8_a30k_k3_25k_locked/model_best.pt"
    ["J_svdk5_a30k"]="ft_theorem8_a30k_k5_restart_locked|results/ft_theorem8_a30k_k5_restart_locked/model_best.pt"
    ["K_a100k_lr1e5"]="ft_locked_a100k_lr1e5_25k_locked|results/ft_locked_a100k_lr1e5_25k_locked/model_best.pt"
    ["L_a100k_lr1e6"]="ft_locked_a100k_lr1e6_25k_locked|results/ft_locked_a100k_lr1e6_25k_locked/model_best.pt"
    ["M_a300k_lr1e5"]="ft_locked_a300k_lr1e5_25k_locked|results/ft_locked_a300k_lr1e5_25k_locked/model_best.pt"
    ["N_a300k_lr1e6"]="ft_locked_a300k_lr1e6_25k_locked|results/ft_locked_a300k_lr1e6_25k_locked/model_best.pt"
)

OUT_CSV="$SCRIPT_DIR/clean_ppl_results.csv"
echo "label,ckpt_name,val_loss,val_ppl,test_set" > "$OUT_CSV"

for label in "${!CKPTS[@]}"; do
    IFS='|' read -r ckpt_name ckpt_path <<< "${CKPTS[$label]}"
    echo ""
    echo "=== $label ($ckpt_name) ==="

    if [ "$ckpt_name" = "pretrained" ]; then
        echo "[SKIP] Pretrained — PPL already evaluated on original held-out."
        # We'll still need to re-evaluate pretrained on the final-test set
        # Fall through to evaluation with --ckpt_name pretrained --ckpt_path pretrained
    fi

    # Use a small wrapper to compute PPL on the final_test.fasta
    echo "  Computing PPL on final_test.fasta..."
    # The existing attack_ppl.py takes --ckpts_csv with a custom CSV
    # We create a temp CSV pointing to final_test.fasta
    TMP_CSV=$(mktemp --suffix=.csv)
    echo "name,ckpt" > "$TMP_CSV"
    echo "${ckpt_name},${ckpt_path}" >> "$TMP_CSV"

    CUDA_VISIBLE_DEVICES="$GPU" $PYTHON scripts/attack_ppl.py \
        --n_batches 64 \
        --ckpts_csv "$TMP_CSV" \
        --out /tmp/clean_ppl_tmp.csv \
        --data_path "$TEST_FASTA" \
        2>&1 | tail -3

    # Extract result and append to clean CSV
    if [ -f /tmp/clean_ppl_tmp.csv ]; then
        tail -n +2 /tmp/clean_ppl_tmp.csv | while IFS=',' read -r name ckpt val_loss val_ppl; do
            echo "$label,$name,$val_loss,$val_ppl,final_test" >> "$OUT_CSV"
        done
    fi
    rm -f "$TMP_CSV" /tmp/clean_ppl_tmp.csv
done

echo ""
echo "=== Done. Results: $OUT_CSV ==="
cat "$OUT_CSV"
