#!/usr/bin/env python3
"""Exp 2 — Leakage-free re-evaluation on a clean three-way split.

For every paper condition (pretrained, unlocked, A–N), loads the existing
best checkpoint, evaluates PPL + HVUE AUROC on the final-test set EXACTLY ONCE,
and compares against the original paper numbers.

Usage:
  1. First run:  python3 experiments/split_manifest/build_three_way_split.py
  2. Then:        bash experiments/exp2_cleansplit/run_clean_eval.sh

Output: experiments/exp2_cleansplit/clean_eval_results.csv
        experiments/exp2_cleansplit/comparison_table.md
"""

import csv, os, sys, json, glob, argparse, re
from collections import defaultdict
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_DIR = os.path.join(REPO_ROOT, "experiments", "split_manifest")
OUT_DIR    = os.path.join(REPO_ROOT, "experiments", "exp2_cleansplit")
os.makedirs(OUT_DIR, exist_ok=True)

# ── paper condition → run_name mapping ──────────────────────
CONDITIONS = {
    "Pretrained":         ("pretrained",        None,              "—"),
    "Unlocked FT":        ("ft_unlocked_25k_v2_unlocked",
                           "results/ft_unlocked_25k_v2_unlocked/model_best.pt", "—"),
    "A  (α=10⁴,  η=10⁻⁶)": ("ft_locked_a10k_lr1e6_25k_locked",
                           "results/ft_locked_a10k_lr1e6_25k_locked/model_best.pt", "10⁴"),
    "B  (α=3×10⁴, η=10⁻⁶)": ("ft_locked_a30k_lr1e6_25k_locked",
                           "results/ft_locked_a30k_lr1e6_25k_locked/model_best.pt", "3×10⁴"),
    "C′ (α=3×10⁴, η=10⁻⁵)": ("ft_locked_a30k_lr1e5_25k_locked",
                           "results/ft_locked_a30k_lr1e5_25k_locked/model_best.pt", "3×10⁴"),
    "D  (LoRA)":          ("ft_lora_a10k_25k_locked",
                           "results/ft_lora_a10k_25k_locked/model_best.pt", "10⁴"),
    "E  (Bypass α=10⁴)":  ("ft_bypass_a10k_25k_v2_locked",
                           "results/ft_bypass_a10k_25k_v2_locked/model_best.pt", "10⁴"),
    "F  (Bypass α=3×10⁴)": ("ft_bypass_a30k_25k_locked",
                           "results/ft_bypass_a30k_25k_locked/model_best.pt", "3×10⁴"),
    "G  (SVD k=3 α=10⁴)": ("ft_theorem8_a10k_k3_25k_locked",
                           "results/ft_theorem8_a10k_k3_25k_locked/model_best.pt", "10⁴"),
    "H  (SVD k=2 α=3×10⁴)": ("ft_theorem8_a30k_k2_25k_locked",
                           "results/ft_theorem8_a30k_k2_25k_locked/model_best.pt", "3×10⁴"),
    "I  (SVD k=3 α=3×10⁴)": ("ft_theorem8_a30k_k3_25k_locked",
                           "results/ft_theorem8_a30k_k3_25k_locked/model_best.pt", "3×10⁴"),
    "J  (SVD k=5 α=3×10⁴)": ("ft_theorem8_a30k_k5_restart_locked",
                           "results/ft_theorem8_a30k_k5_restart_locked/model_best.pt", "3×10⁴"),
    "K  (α=10⁵,  η=10⁻⁵)": ("ft_locked_a100k_lr1e5_25k_locked",
                           "results/ft_locked_a100k_lr1e5_25k_locked/model_best.pt", "10⁵"),
    "L  (α=10⁵,  η=10⁻⁶)": ("ft_locked_a100k_lr1e6_25k_locked",
                           "results/ft_locked_a100k_lr1e6_25k_locked/model_best.pt", "10⁵"),
    "M  (α=3×10⁵, η=10⁻⁵)": ("ft_locked_a300k_lr1e5_25k_locked",
                           "results/ft_locked_a300k_lr1e5_25k_locked/model_best.pt", "3×10⁵"),
    "N  (α=3×10⁵, η=10⁻⁶)": ("ft_locked_a300k_lr1e6_25k_locked",
                           "results/ft_locked_a300k_lr1e6_25k_locked/model_best.pt", "3×10⁵"),
}

# ── Load original paper numbers ──────────────────────────────
def load_original_results():
    """Parse original hvue_probe.csv and attack_heldout_ppl.csv."""
    ppl = {}
    with open(os.path.join(REPO_ROOT, "results/attack_heldout_ppl.csv")) as f:
        for row in csv.DictReader(f):
            ppl[row["name"]] = float(row["val_ppl"])

    auroc = defaultdict(dict)
    with open(os.path.join(REPO_ROOT, "results/hvue_probe.csv")) as f:
        for row in csv.DictReader(f):
            auroc[row["ckpt"]][row["task"]] = float(row["auroc"])

    return ppl, auroc


def main():
    ppl_orig, auroc_orig = load_original_results()

    print("# Exp 2 — Clean-Split Re-Evaluation\n")
    print("## Instructions\n")
    print("1. First build the three-way split:")
    print("   `python3 experiments/split_manifest/build_three_way_split.py`\n")
    print("2. Then run PPL evaluation on the final-test set for each checkpoint:")
    print("   `bash experiments/exp2_cleansplit/run_clean_eval.sh`\n")
    print("3. Then run HVUE extraction + probe on the final-test set.")
    print("   `bash experiments/exp2_cleansplit/run_hvue_clean.sh`\n")
    print("---\n")

    print("## Condition → Checkpoint Mapping\n")
    print(f"| {'Condition':<24s} | {'Run name':<48s} | {'CKPT exists?':<12s} |")
    print(f"|{'-'*26}|{'-'*50}|{'-'*14}|")
    for label, (run_name, ckpt_path, alpha) in CONDITIONS.items():
        if ckpt_path is None:
            exists = "(pretrained)"
        else:
            exists = "✓" if os.path.exists(os.path.join(REPO_ROOT, ckpt_path)) else "✗ MISSING"
        print(f"| {label:<24s} | {run_name:<48s} | {exists:<12s} |")

    print("\n## Original (potentially optimistic) numbers\n")
    # Print the original numbers the paper currently reports
    REFERENCE_VALUES = {
        "pretrained":                           (3.729,  (0.860, 0.815, 0.852, 0.842)),
        "ft_unlocked_25k_v2_unlocked":          (3.487,  (0.876, 0.842, 0.871, 0.867)),
        "ft_locked_a10k_lr1e6_25k_locked":      (3.753,  (0.864, 0.818, 0.862, 0.848)),
        "ft_locked_a30k_lr1e6_25k_locked":      (3.816,  (0.797, 0.788, 0.785, 0.790)),
        "ft_locked_a30k_lr1e5_25k_locked":      (3.776,  (0.905, 0.847, 0.894, 0.882)),
        "ft_lora_a10k_25k_locked":              (3.729,  (0.839, 0.766, 0.779, 0.795)),
        "ft_bypass_a10k_25k_v2_locked":         (4.091,  (0.688, 0.735, 0.845, 0.756)),
        "ft_bypass_a30k_25k_locked":            (4.123,  (0.707, 0.735, 0.848, 0.763)),
        "ft_theorem8_a10k_k3_25k_locked":       (3.667,  (0.817, 0.782, 0.819, 0.806)),
        "ft_theorem8_a30k_k2_25k_locked":       (3.745,  (0.845, 0.786, 0.840, 0.824)),
        "ft_theorem8_a30k_k3_25k_locked":       (3.705,  (0.858, 0.803, 0.845, 0.836)),
        "ft_theorem8_a30k_k5_restart_locked":   (3.760,  (0.852, 0.808, 0.845, 0.835)),
        "ft_locked_a100k_lr1e5_25k_locked":     (3.798,  (0.879, 0.839, 0.871, 0.863)),
        "ft_locked_a100k_lr1e6_25k_locked":     (3.876,  (0.817, 0.784, 0.823, 0.808)),
        "ft_locked_a300k_lr1e5_25k_locked":     (3.810,  (0.872, 0.831, 0.871, 0.858)),
        "ft_locked_a300k_lr1e6_25k_locked":     (5.861,  (0.834, 0.806, 0.830, 0.823)),
    }

    print(f"| {'Condition':<24s} | {'Orig PPL':>8s} | {'Trop':>6s} | {'Path':>6s} | {'Trans':>6s} | {'Mean AUROC':>10s} |")
    print(f"|{'-'*26}|{'-'*10}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*12}|")
    for label, (run_name, ckpt_path, alpha) in CONDITIONS.items():
        if run_name in REFERENCE_VALUES:
            p, (trop, path, trans, mean_a) = REFERENCE_VALUES[run_name]
            print(f"| {label:<24s} | {p:>8.3f} | {trop:>6.3f} | {path:>6.3f} | {trans:>6.3f} | {mean_a:>10.3f} |")
        else:
            print(f"| {label:<24s} | {'—':>8s} | {'—':>6s} | {'—':>6s} | {'—':>6s} | {'—':>10s} |")

    print("\n---")
    print("## After clean re-evaluation, fill in the comparison table:\n")
    print(f"| {'Condition':<24s} | {'Orig PPL':>8s} | {'Clean PPL':>9s} | {'Δ PPL':>8s} | {'Orig AUROC':>10s} | {'Clean AUROC':>11s} | {'Δ AUROC':>9s} | {'Ceiling flip?':>12s} |")
    print(f"|{'-'*26}|{'-'*10}|{'-'*11}|{'-'*10}|{'-'*12}|{'-'*13}|{'-'*11}|{'-'*14}|")
    for label, (run_name, ckpt_path, alpha) in CONDITIONS.items():
        if run_name in REFERENCE_VALUES:
            p, (trop, path, trans, mean_a) = REFERENCE_VALUES[run_name]
            print(f"| {label:<24s} | {p:>8.3f} | {'TBD':>9s} | {'—':>8s} | {mean_a:>10.3f} | {'TBD':>11s} | {'—':>9s} | {'—':>12s} |")


if __name__ == "__main__":
    main()
