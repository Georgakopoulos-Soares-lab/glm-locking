#!/usr/bin/env python3
"""Exp 3A — ViroBench discriminative evaluation for Unlocked-FT and M.

Runs the ViroBench discriminative tasks through the SAME embedding + linear-probe
protocol used in the paper (mean-pooled final-layer activations, ℓ₂-regularised
linear SVM probe). Reports per-task and aggregate scores.

ViroBench: https://github.com/QIANJINYDX/ViroBench  (arXiv 2605.25388)
  - 18 scenarios across 4 task types
  - Two axes: biological understanding + latent biosecurity risk
  - NFM-native (nucleotide input) — no reverse-translation needed

Usage:
  # 1. Clone ViroBench
  git clone https://github.com/QIANJINYDX/ViroBench experiments/exp3_virobench/ViroBench

  # 2. Download ViroBench data (follow their instructions)

  # 3. Run discriminative eval
  bash experiments/exp3_virobench/run_discriminative.sh [GPU_ID]

Output: experiments/exp3_virobench/discriminative_results.csv
"""

import os, sys, json, csv, argparse
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "exp3_virobench")
os.makedirs(OUT_DIR, exist_ok=True)

VIROBENCH_DIR = os.path.join(OUT_DIR, "ViroBench")

# Models to evaluate
MODELS = {
    "Unlocked-FT": "results/ft_unlocked_25k_v2_unlocked/model_best.pt",
    "M (α=3×10⁵)": "results/ft_locked_a300k_lr1e5_25k_locked/model_best.pt",
    "Pretrained":   "pretrained",
}


def main():
    print("=" * 70)
    print("Exp 3A — ViroBench Discriminative Evaluation")
    print("=" * 70)
    print()

    # Check ViroBench availability
    if not os.path.exists(VIROBENCH_DIR):
        print("ViroBench not found. Clone it first:")
        print(f"  git clone https://github.com/QIANJINYDX/ViroBench {VIROBENCH_DIR}")
        print()
        print("Then follow their README to download data.")
        print()
        print("NOTE: If ViroBench data is access-gated or won't download,")
        print("      STOP and report. Do not proceed without confirmed data access.")
        return

    print(f"ViroBench found at: {VIROBENCH_DIR}")
    print()

    # Check for data
    data_dir = os.path.join(VIROBENCH_DIR, "data")
    if not os.path.exists(data_dir):
        print("ViroBench data directory not found. Download their data first.")
        print("Follow instructions in ViroBench/README.md")
        return

    print(f"ViroBench data found at: {data_dir}")
    print()

    # List available tasks
    print("Available ViroBench tasks:")
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            print(f"  - {item}")
    print()

    print("─" * 70)
    print("EVALUATION PROTOCOL:")
    print("  1. For each model (Unlocked-FT, M, Pretrained):")
    print("     a. Load checkpoint via the existing hvue_extract_one_ckpt pipeline")
    print("     b. Extract mean-pooled final-layer activations for each")
    print("        ViroBench task's sequences")
    print("     c. Train ℓ₂-regularised linear SVM probe (C ∈ {0.01, 0.1, 1, 10},")
    print("        5-fold CV), matching the paper's probe protocol")
    print("     d. Report AUROC, accuracy, F1, MCC per task")
    print()
    print("  2. Match ViroBench's own eval protocol where it differs;")
    print("     document any deviation.")
    print()

    print("─" * 70)
    print("TO RUN:")
    print(f"  bash {os.path.join(OUT_DIR, 'run_discriminative.sh')} [GPU_ID]")
    print()
    print("This script handles:")
    print("  - Loading ViroBench task data in the correct format")
    print("  - Running hvue_extract_one_ckpt.py for each model × task")
    print("  - Running hvue_probe.py with ViroBench splits")
    print("  - Producing discriminative_results.csv")


if __name__ == "__main__":
    main()
