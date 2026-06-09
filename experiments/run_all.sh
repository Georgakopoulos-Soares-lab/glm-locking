#!/bin/bash
# Master script — run all three follow-up experiments in recommended order.
#
# RECOMMENDED ORDER:
#   1. Exp 2 (cheapest, defines the clean split everything else uses)
#   2. Exp 3 Part A (discriminative; reuses checkpoints + clean split)
#   3. Exp 3 Part B (generative; novel; needs geNomad/Prodigal/BLAST)
#   4. Exp 1 (most expensive; two full 10k-scale training runs)
#
# Usage:
#   bash experiments/run_all.sh [GPU_ID]

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${1:-0}"
EXPERIMENTS="$REPO_ROOT/experiments"

echo "============================================================"
echo "  SpecDef Locking — Follow-Up Experiments"
echo "============================================================"
echo ""
echo "GPU: $GPU"
echo "Recommended execution order: Exp 2 → Exp 3A → Exp 3B → Exp 1"
echo ""

# ── STOP-AND-ASK conditions ──────────────────────────────────
echo "BEFORE STARTING, confirm:"
echo "  [ ] ViroBench data is NOT access-gated (check Exp 3A)"
echo "  [ ] geNomad, Pyrodigal, BLAST are installed (for Exp 3B)"
echo "  [ ] ~60h GPU budget available for Exp 1 (two 10k-scale runs)"
echo "  [ ] NCBI Virus access works for 10k corpus assembly"
echo ""
read -p "Press Enter to continue, or Ctrl-C to abort... " _

# ── Step 0: Build three-way split ────────────────────────────
echo ""
echo "=== STEP 0: Build three-way split ==="
cd "$REPO_ROOT"
python3 experiments/split_manifest/build_three_way_split.py \
    --input data/attack.fasta \
    --train-frac 0.70 --val-frac 0.15 --test-frac 0.15 \
    --seed 42

# ── Experiment 2: Clean re-evaluation ────────────────────────
echo ""
echo "=== EXPERIMENT 2: Clean re-evaluation ==="
echo "PPL evaluation on final-test set..."
bash experiments/exp2_cleansplit/run_clean_eval.sh "$GPU"
echo "Exp 2 PPL done. Next: HVUE extraction + probe on final-test set."
echo "  (Manual step — run hvue_extract_one_ckpt for each checkpoint"
echo "   using final_test.fasta as input, then probe.)"

# ── Experiment 3A: ViroBench discriminative ──────────────────
echo ""
echo "=== EXPERIMENT 3A: ViroBench discriminative ==="
if [ -d "$EXPERIMENTS/exp3_virobench/ViroBench" ]; then
    echo "ViroBench found."
    echo "Run: bash experiments/exp3_virobench/run_discriminative.sh $GPU"
else
    echo "ViroBench not cloned. Clone it first:"
    echo "  git clone https://github.com/QIANJINYDX/ViroBench $EXPERIMENTS/exp3_virobench/ViroBench"
    echo "STOP-AND-ASK: If ViroBench data is access-gated, do not proceed."
fi

# ── Experiment 3B: Generative evaluation ─────────────────────
echo ""
echo "=== EXPERIMENT 3B: Generative evaluation ==="
echo "Step 1 (generation): bash experiments/exp3_virobench/partB_generative/run_generation.sh $GPU"
echo "Step 2 (scoring):    bash experiments/exp3_virobench/partB_generative/run_scoring.sh"
echo "Step 3 (comparison):  python3 experiments/exp3_virobench/partB_generative/compare_distributions.py"

# ── Experiment 1: 10k data scale ─────────────────────────────
echo ""
echo "=== EXPERIMENT 1: 10k data scale ==="
echo "Step 1: Assemble 10k corpus"
echo "  python3 experiments/exp1_datascale/assemble_10k_corpus.py"
echo "Step 2: Three-way split the 10k corpus"
echo "  python3 experiments/split_manifest/build_three_way_split.py \\"
echo "      --input experiments/exp1_datascale/attack_10k.fasta \\"
echo "      --train-frac 0.70 --val-frac 0.15 --test-frac 0.15 \\"
echo "      --seed 42 \\"
echo "      --out-dir experiments/exp1_datascale/split"
echo "Step 3: Train unlocked at 10k scale (~31h)"
echo "  bash scripts/run_pipeline.sh $GPU experiments/exp1_datascale/unlocked_10k_25k.yaml"
echo "Step 4: Train M (α=3×10⁵) at 10k scale (~31h)"
echo "  bash scripts/run_pipeline.sh $GPU experiments/exp1_datascale/locked_a300k_lr1e5_10k_25k.yaml"
echo "Step 5: Evaluate both on final-test set"

echo ""
echo "============================================================"
echo "  Ready. Follow the instructions above for each experiment."
echo "============================================================"
