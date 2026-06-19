"""Probe-depth sweep: LR vs MLP probes on frozen HVUE embeddings.

Tests whether a stronger classifier (MLP) extracts more signal from frozen
embeddings than logistic regression. Compares both to k-mer composition baseline.

Key checkpoints: pretrained, unlocked-FT, M (α=3×10⁵), K (α=10⁵),
SVD-best-1 (α=3×10⁴, k=1), SVD-best-2 (α=10⁴, k=3).
"""
from __future__ import annotations
import os, sys, csv, glob, json
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

EMB_DIR = "results/hvue_embeddings"
OUT_CSV = "results/hvue_probe_depth_sweep.csv"

# Key checkpoints to evaluate
CKPTS = [
    "pretrained",
    "ft_unlocked_25k_v2_unlocked",  # unlocked-FT
    "ft_locked_a300k_lr1e5_25k_locked",  # M (α=3×10⁵)
    "ft_locked_a100k_lr1e5_25k_locked",  # K (α=10⁵)
    "ft_theorem8_a30k_k1_25k_locked",    # SVD best 1
    "ft_theorem8_a10k_k3_25k_locked",    # SVD best 2
]

TASKS = ["Host_Tropism", "Pathogenecity", "Transmissibility"]

# k-mer composition baseline (from hvue_composition_audit)
KMER_BASELINE = {
    "Host_Tropism":      0.8705,  # k=3
    "Pathogenecity":     0.8416,  # k=4
    "Transmissibility":  0.8963,  # k=4
}


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d['X'], d['y']


def probe_lr(X_tr, y_tr, X_va, y_va, C_values=[0.01, 0.1, 1.0, 10.0], seed=0):
    """Logistic regression probe (same as hvue_probe.py)."""
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_va_s = sc.transform(X_va)

    best = None
    best_clf = None
    for C in C_values:
        clf = LogisticRegression(max_iter=2000, C=C, random_state=seed, n_jobs=-1)
        clf.fit(X_tr_s, y_tr)
        p = clf.predict_proba(X_va_s)[:, 1]
        try:
            auroc = roc_auc_score(y_va, p)
        except Exception:
            auroc = float('nan')
        if best is None or auroc > best[0]:
            best = (auroc, C)
            best_clf = clf

    auroc, C = best
    clf = best_clf
    pred = clf.predict(X_va_s)
    return dict(
        auroc=float(auroc),
        acc=float(accuracy_score(y_va, pred)),
        f1=float(f1_score(y_va, pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_va, pred)),
        best_C=float(C),
        probe_type="LR",
    )


def probe_mlp(X_tr, y_tr, X_va, y_va, hidden=256, seed=0):
    """2-layer MLP probe with ReLU — tests if deeper head extracts more signal."""
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_va_s = sc.transform(X_va)

    best = None
    best_clf = None
    for alpha in [0.0001, 0.001, 0.01]:
        for lr in [0.001, 0.0005]:
            clf = MLPClassifier(
                hidden_layer_sizes=(hidden, hidden),
                activation='relu',
                alpha=alpha,
                learning_rate_init=lr,
                max_iter=500,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )
            clf.fit(X_tr_s, y_tr)
            p = clf.predict_proba(X_va_s)[:, 1]
            try:
                auroc = roc_auc_score(y_va, p)
            except Exception:
                auroc = float('nan')
            if best is None or auroc > best[0]:
                best = (auroc, alpha, lr)
                best_clf = clf

    auroc, alpha, lr = best
    clf = best_clf
    pred = clf.predict(X_va_s)
    return dict(
        auroc=float(auroc),
        acc=float(accuracy_score(y_va, pred)),
        f1=float(f1_score(y_va, pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_va, pred)),
        best_alpha=float(alpha),
        best_lr=float(lr),
        probe_type="MLP",
    )


def main():
    print("=" * 70)
    print("Probe-Depth Sweep: LR vs MLP on Frozen HVUE Embeddings")
    print("=" * 70)

    rows = []

    for ckpt in CKPTS:
        print(f"\n--- {ckpt} ---")
        for task in TASKS:
            tr_path = f"{EMB_DIR}/{ckpt}_{task}_train.npz"
            va_path = f"{EMB_DIR}/{ckpt}_{task}_validation.npz"

            if not os.path.exists(tr_path) or not os.path.exists(va_path):
                print(f"  {task}: MISSING embeddings")
                continue

            X_tr, y_tr = load_npz(tr_path)
            X_va, y_va = load_npz(va_path)

            # Logistic regression
            r_lr = probe_lr(X_tr, y_tr, X_va, y_va)
            r_lr['ckpt'] = ckpt
            r_lr['task'] = task
            rows.append(r_lr)

            # MLP probe
            r_mlp = probe_mlp(X_tr, y_tr, X_va, y_va)
            r_mlp['ckpt'] = ckpt
            r_mlp['task'] = task
            rows.append(r_mlp)

            kmer_auroc = KMER_BASELINE.get(task, float('nan'))
            lr_delta = r_lr['auroc'] - kmer_auroc
            mlp_delta = r_mlp['auroc'] - kmer_auroc

            print(f"  {task:20s}  LR={r_lr['auroc']:.4f} (Δkmer={lr_delta:+.4f})  "
                  f"MLP={r_mlp['auroc']:.4f} (Δkmer={mlp_delta:+.4f})  "
                  f"kmer={kmer_auroc:.4f}")

    # Save
    if rows:
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with open(OUT_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                for k in keys:
                    if k not in r:
                        r[k] = ''
            w.writerows(rows)
        print(f"\nSaved: {OUT_CSV}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Best probe AUROC vs k-mer baseline")
    print("=" * 70)
    print(f"{'Checkpoint':35s} {'Task':20s} {'LR':8s} {'MLP':8s} {'k-mer':8s} {'Best Δ':8s}")
    print("-" * 85)

    for ckpt in CKPTS:
        for task in TASKS:
            ckpt_rows = [r for r in rows if r['ckpt'] == ckpt and r['task'] == task]
            if not ckpt_rows:
                continue
            lr = max(r['auroc'] for r in ckpt_rows if r['probe_type'] == 'LR')
            mlp = max(r['auroc'] for r in ckpt_rows if r['probe_type'] == 'MLP')
            kmer = KMER_BASELINE.get(task, 0)
            best = max(lr, mlp)
            delta = best - kmer
            flag = "⚠ CONFOUND" if delta < 0.03 else "✓ ABOVE KMER"
            print(f"{ckpt:35s} {task:20s} {lr:.4f}  {mlp:.4f}  {kmer:.4f}  {delta:+.4f}  {flag}")

    print(f"\nk-mer baseline values: {KMER_BASELINE}")
    print("Δ < 0.03 → task signal is indistinguishable from composition → CONFOUND")


if __name__ == "__main__":
    main()
