"""Bootstrap CI and pairwise significance for HVUE LoRA results.

Two routes for pairwise comparisons:
A) If per-sample val predictions saved (results/hvue_lora_preds/*.npz):
   - Paired bootstrap AUROC CI per run
   - Paired bootstrap Δ for model comparisons (more powerful, controls
     for shared val-set variance)

B) Fallback (Hanley-McNeil + independent z-test) for runs without saved predictions:
   - HM variance formula gives SE(AUROC) from AUROC + n_pos + n_neg alone
   - Two-sided independent-samples z-test: z = Δ / sqrt(SE_a² + SE_b²)
   - Conservative (ignores positive correlation from shared val set)
   - Valid null-hypothesis test; ΔCI = Δ ± 1.96·SE_diff

Pairwise comparisons reported:
  1. locked_no_ft − pretrained  (path artifact: does lock obstruct LoRA learning?)
  2. unlocked_ft − pretrained   (FT capability boost)
  3. M_a300k − unlocked_ft      (lock defense: does attack model learn less?)
  4. M_a300k − pretrained       (net: lock+FT vs base)
  5. each model − k-mer         (beats composition baseline?)

Run:
  python scripts/lora_bootstrap_analysis.py \
      --csv results/hvue_lora_all.csv \
      --preds_dir results/hvue_lora_preds \
      --out results/hvue_lora_ci.csv \
      --out_pairs results/hvue_lora_pairwise.csv
"""
from __future__ import annotations
import os, sys, argparse, glob, itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BOOT = 10000
SEED   = 0

KMER_AUCS = {
    "Host_Tropism":     0.9030,
    "Pathogenecity":    0.8459,
    "Transmissibility": 0.9137,
}

PAIR_DEFS = [
    ("path_artifact", "locked_no_ft", "pretrained"),
    ("ft_effect",     "unlocked_ft",  "pretrained"),
    ("lock_effect",   "M_a300k",      "unlocked_ft"),
    ("lock_vs_base",  "M_a300k",      "pretrained"),
]

TASKS = ["Host_Tropism", "Pathogenecity", "Transmissibility"]
CKPTS = ["pretrained", "locked_no_ft", "unlocked_ft", "M_a300k"]


# ── Hanley-McNeil closed-form CI (no score vector needed) ────────────────────

def hanley_mcneil_se(auroc: float, n_pos: int, n_neg: int) -> float:
    """SE(AUROC) from Hanley-McNeil (1983)."""
    A = auroc
    Q1 = A / (2 - A)
    Q2 = 2 * A**2 / (1 + A)
    var = (A * (1 - A)
           + (n_pos - 1) * (Q1 - A**2)
           + (n_neg - 1) * (Q2 - A**2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0)))


def hanley_mcneil_ci(auroc: float, n_pos: int, n_neg: int) -> tuple[float, float]:
    se = hanley_mcneil_se(auroc, n_pos, n_neg)
    return float(auroc - 1.96 * se), float(auroc + 1.96 * se)


def hm_independent_ztest(auroc_a: float, auroc_b: float, n_pos: int, n_neg: int
                         ) -> tuple[float, float, float, str]:
    """Independent-samples z-test for AUROC_a − AUROC_b.
    Conservative because it ignores positive correlation from shared val set.
    Returns: obs_delta, delta_ci_lo, delta_ci_hi, p_value, significance.
    """
    se_a = hanley_mcneil_se(auroc_a, n_pos, n_neg)
    se_b = hanley_mcneil_se(auroc_b, n_pos, n_neg)
    se_diff = np.sqrt(se_a**2 + se_b**2)
    delta = auroc_a - auroc_b
    z = delta / se_diff if se_diff > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    lo = float(delta - 1.96 * se_diff)
    hi = float(delta + 1.96 * se_diff)
    sig = "sig" if p < 0.05 else "ns"
    return delta, lo, hi, p, sig


# ── Bootstrap from saved predictions ─────────────────────────────────────────

def boot_auroc_ci(preds: np.ndarray, labels: np.ndarray, n_boot=N_BOOT, seed=SEED):
    """Bootstrap 95% CI for AUROC from saved per-sample predictions."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = labels[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(roc_auc_score(yb, preds[idx]))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_paired_delta(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray,
                      n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap of Δ = AUROC(a) − AUROC(b) on the SAME val set."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = labels[idx]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(roc_auc_score(yb, preds_a[idx]) - roc_auc_score(yb, preds_b[idx]))
    deltas = np.array(deltas)
    point = float(np.mean(deltas))
    lo    = float(np.percentile(deltas, 2.5))
    hi    = float(np.percentile(deltas, 97.5))
    p     = float(np.mean(deltas <= 0) if point > 0 else np.mean(deltas >= 0))
    p     = float(min(2 * p, 1.0))   # two-sided
    sig   = "sig" if p < 0.05 else "ns"
    return point, lo, hi, p, sig


# ── Load predictions ──────────────────────────────────────────────────────────

def load_preds(preds_dir: str, task: str, ckpt: str, lr: float) -> tuple | None:
    key = f"{task}_{ckpt}_{lr:.0e}"
    path = os.path.join(preds_dir, f"{key}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["preds"], d["labels"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       default="results/hvue_lora_all.csv")
    ap.add_argument("--preds_dir", default="results/hvue_lora_preds")
    ap.add_argument("--out",       default="results/hvue_lora_ci.csv")
    ap.add_argument("--out_pairs", default="results/hvue_lora_pairwise.csv")
    ap.add_argument("--n_boot",    type=int, default=N_BOOT)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    # Use best-LR row per (task, ckpt)
    best = df.loc[df.groupby(["task", "ckpt"])["best_val_auroc"].idxmax()].copy()

    ci_rows   = []
    pair_rows = []

    for task in TASKS:
        t_rows = best[best.task == task]
        if len(t_rows) == 0:
            continue
        kmer_auc = KMER_AUCS[task]
        # Assume balanced val: n_pos = n_neg = N_VAL//2 = 1000
        n_pos = n_neg = 1000

        # ── Per-checkpoint CI ────────────────────────────────────────────────
        task_preds:  dict[str, tuple] = {}
        task_aurocs: dict[str, float] = {}
        for _, row in t_rows.iterrows():
            ckpt  = row["ckpt"]
            lr    = row["lr"]
            auroc = row["best_val_auroc"]
            task_aurocs[ckpt] = auroc

            preds_data = load_preds(a.preds_dir, task, ckpt, lr)
            if preds_data is not None:
                preds, labels = preds_data
                n_pos_real = int(labels.sum())
                n_neg_real = len(labels) - n_pos_real
                mean_a, lo, hi = boot_auroc_ci(preds, labels, n_boot=a.n_boot)
                method = "bootstrap"
                task_preds[ckpt] = (preds, labels)
            else:
                lo, hi = hanley_mcneil_ci(auroc, n_pos, n_neg)
                method = "hanley-mcneil"
                n_pos_real, n_neg_real = n_pos, n_neg

            # model vs k-mer z-test (independent HM; conservative)
            kmer_delta, kd_lo, kd_hi, kmer_p, kmer_sig = hm_independent_ztest(
                auroc, kmer_auc, n_pos, n_neg)
            klo, khi = hanley_mcneil_ci(kmer_auc, n_pos, n_neg)

            ci_rows.append(dict(
                task=task, ckpt=ckpt, lr=lr,
                best_val_auroc=round(auroc, 4),
                ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                ci_method=method,
                kmer_auroc=round(kmer_auc, 4),
                kmer_delta=round(kmer_delta, 4),
                kmer_delta_ci_lo=round(kd_lo, 4),
                kmer_delta_ci_hi=round(kd_hi, 4),
                kmer_p=round(kmer_p, 4),
                kmer_sig=kmer_sig,
                n_val=n_pos_real + n_neg_real,
            ))
            print(f"  [{task}] {ckpt:14s}  AUROC={auroc:.4f}  CI=[{lo:.4f},{hi:.4f}]  "
                  f"vs k-mer Δ={kmer_delta:+.4f} [{kd_lo:+.4f},{kd_hi:+.4f}] "
                  f"p={kmer_p:.4f} {kmer_sig}  ({method})")

        # ── Pairwise model comparisons ────────────────────────────────────────
        print(f"\n  [{task}] pairwise Δ:")
        for label, a_name, b_name in PAIR_DEFS:
            if a_name not in task_aurocs or b_name not in task_aurocs:
                print(f"    {label:14s} ({a_name}−{b_name}): SKIP — AUROC not available")
                continue
            auroc_a = task_aurocs[a_name]
            auroc_b = task_aurocs[b_name]
            obs_delta = auroc_a - auroc_b

            # Use paired bootstrap if both have saved predictions
            if a_name in task_preds and b_name in task_preds:
                pa, la = task_preds[a_name]
                pb, lb = task_preds[b_name]
                if np.array_equal(la, lb):
                    point, lo, hi, p, sig = boot_paired_delta(pa, pb, la, n_boot=a.n_boot)
                    method = "paired-bootstrap"
                else:
                    _, lo, hi, p, sig = hm_independent_ztest(auroc_a, auroc_b, n_pos, n_neg)
                    method = "hm-indep-z (label mismatch)"
            else:
                _, lo, hi, p, sig = hm_independent_ztest(auroc_a, auroc_b, n_pos, n_neg)
                method = "hm-indep-z"

            pair_rows.append(dict(
                task=task, comparison=label, a=a_name, b=b_name,
                obs_delta=round(obs_delta, 4),
                ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                p_value=round(p, 4),
                significant=sig,
                test_method=method,
            ))
            print(f"    {label:14s} ({a_name}−{b_name}): Δ={obs_delta:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] p={p:.4f} {sig}  ({method})")
        print()

    # Save
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(ci_rows).to_csv(a.out, index=False)
    print(f"[done] CI table → {a.out}")
    pd.DataFrame(pair_rows).to_csv(a.out_pairs, index=False)
    print(f"[done] pairwise table → {a.out_pairs}")

    # ── Summary tables ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: best-LR AUROC with 95% CI and model-vs-kmer test")
    print("=" * 70)
    ci_df = pd.DataFrame(ci_rows)
    for task in TASKS:
        t = ci_df[ci_df.task == task]
        if len(t) == 0:
            continue
        k = KMER_AUCS[task]
        klo_g, khi_g = hanley_mcneil_ci(k, 1000, 1000)
        print(f"\n[{task}]  k-mer = {k:.4f}  [{klo_g:.4f},{khi_g:.4f}]")
        print(t[["ckpt","best_val_auroc","ci_lo","ci_hi",
                  "kmer_delta","kmer_p","kmer_sig"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("PAIRWISE Δ between checkpoints with 95% CI and significance")
    print("=" * 70)
    pf = pd.DataFrame(pair_rows)
    for task in TASKS:
        t = pf[pf.task == task]
        if len(t):
            print(f"\n[{task}]")
            print(t[["comparison","a","b","obs_delta","ci_lo","ci_hi",
                      "p_value","significant","test_method"]].to_string(index=False))


if __name__ == "__main__":
    main()
