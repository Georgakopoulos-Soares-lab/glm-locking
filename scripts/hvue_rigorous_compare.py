"""Rigorous, matched model-vs-composition AUROC comparison for HVUE.

Fixes the methodological inconsistencies in the original confound audit:
  1. SAME split        — model and k-mer are both trained on `train`, reported on `validation`.
  2. SAME sequences     — k-mer features are computed on the EXACT rows used to extract
                          the model embeddings (extraction sampling is reproduced bit-for-bit).
  3. SAME class balance — both use the balanced (y_mean=0.5) sampled set, not natural prevalence.
  4. NO eval-set leakage— regularisation C is chosen by 5-fold CV on `train`, never on the
                          reported `validation` split (the original audit picked C on the test set).
  5. Residual capability— reports model AUROC, k-mer AUROC, and the residual (model - k-mer)
                          with a paired bootstrap 95% CI, which is the quantity that actually
                          isolates "signal beyond sequence composition".

GPU-free: reuses precomputed embeddings in results/hvue_embeddings/.

Usage:
  python scripts/hvue_rigorous_compare.py \
      --ckpts pretrained ft_unlocked_25k_v2_unlocked ft_locked_a300k_lr1e5_25k_locked \
      --out results/hvue_rigorous_compare.csv
"""
from __future__ import annotations
import os, sys, argparse, itertools, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

HVUE_DIR = "data/hvue"
EMB_DIR = "results/hvue_embeddings"
TASKS = ["Host_Tropism", "Pathogenecity", "Transmissibility"]
N_TRAIN = 3000
N_VAL = 2000
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
NUCLEOTIDES = ['A', 'C', 'G', 'T']
N_BOOT = 5000
SEED = 0


# ── Reproduce the EXACT extraction sampling (hvue_extract_one_ckpt.py) ────────

def repro_sample(task: str, split: str, n: int) -> pd.DataFrame:
    """Reproduce the balanced sample used during embedding extraction, in row order."""
    df = pd.read_parquet(f"{HVUE_DIR}/{task}_{split}.parquet")
    per_class = n // df['label'].nunique()
    sub = df.groupby('label', group_keys=False).apply(
        lambda g: g.sample(min(per_class, len(g)), random_state=0),
        include_groups=False,
    )
    sub = df.loc[sub.index]                       # re-attach label column
    sub = sub.sample(frac=1, random_state=0)      # same shuffle as extractor
    return sub


# ── k-mer composition features ────────────────────────────────────────────────

def kmer_frequencies(seqs, k: int) -> np.ndarray:
    kmers = [''.join(p) for p in itertools.product(NUCLEOTIDES, repeat=k)]
    idx = {km: i for i, km in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s.upper()
        for j in range(len(s) - k + 1):
            km = s[j:j + k]
            if km in idx:
                X[i, idx[km]] += 1
        tot = X[i].sum()
        if tot > 0:
            X[i] /= tot
    return X


# ── Probe with C chosen by CV on TRAIN, reported on VAL ───────────────────────

def fit_probe(X_tr, y_tr, X_va):
    """Return (val_probabilities, chosen_C). C selected by 5-fold CV AUROC on TRAIN only."""
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_va_s = sc.transform(X_va)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_C, best_cv = None, -np.inf
    for C in C_GRID:
        clf = LogisticRegression(max_iter=2000, C=C, random_state=SEED)
        scores = cross_val_score(clf, X_tr_s, y_tr, cv=cv, scoring='roc_auc', n_jobs=-1)
        m = float(np.mean(scores))
        if m > best_cv:
            best_cv, best_C = m, C
    clf = LogisticRegression(max_iter=2000, C=best_C, random_state=SEED).fit(X_tr_s, y_tr)
    p_va = clf.predict_proba(X_va_s)[:, 1]
    return p_va, best_C, best_cv


def boot_residual_ci(y_va, p_model, p_kmer, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap over val samples: distribution of (AUROC_model - AUROC_kmer)."""
    rng = np.random.RandomState(seed)
    n = len(y_va)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y_va[idx]
        if len(np.unique(yb)) < 2:
            continue
        diffs.append(roc_auc_score(yb, p_model[idx]) - roc_auc_score(yb, p_kmer[idx]))
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--kmer_k', type=int, default=4)
    ap.add_argument('--out', default='results/hvue_rigorous_compare.csv')
    a = ap.parse_args()

    # Precompute matched k-mer features + sequences once per task (shared across ckpts).
    print("Computing matched k-mer features (same rows as embeddings)...")
    kmer_cache = {}
    for task in TASKS:
        tr = repro_sample(task, 'train', N_TRAIN)
        va = repro_sample(task, 'validation', N_VAL)
        Xk_tr = kmer_frequencies(tr['sequence'].tolist(), a.kmer_k)
        Xk_va = kmer_frequencies(va['sequence'].tolist(), a.kmer_k)
        kmer_cache[task] = (Xk_tr, tr['label'].values.astype(int),
                            Xk_va, va['label'].values.astype(int))
        print(f"  {task}: kmer{a.kmer_k} feat={Xk_tr.shape[1]}  n_tr={len(tr)} n_va={len(va)}")

    rows = []
    for ck in a.ckpts:
        for task in TASKS:
            etr = f"{EMB_DIR}/{ck}_{task}_train.npz"
            eva = f"{EMB_DIR}/{ck}_{task}_validation.npz"
            if not (os.path.exists(etr) and os.path.exists(eva)):
                print(f"  SKIP {ck}/{task}: missing embeddings")
                continue
            dtr = np.load(etr, allow_pickle=True)
            dva = np.load(eva, allow_pickle=True)
            Xm_tr, ym_tr = dtr['X'], dtr['y'].astype(int)
            Xm_va, ym_va = dva['X'], dva['y'].astype(int)

            Xk_tr, yk_tr, Xk_va, yk_va = kmer_cache[task]
            # Sanity: model and k-mer must be aligned to the same labels/rows.
            assert np.array_equal(ym_tr, yk_tr), f"train label mismatch {ck}/{task}"
            assert np.array_equal(ym_va, yk_va), f"val label mismatch {ck}/{task}"

            p_model, C_m, cv_m = fit_probe(Xm_tr, ym_tr, Xm_va)
            p_kmer, C_k, cv_k = fit_probe(Xk_tr, yk_tr, Xk_va)

            auroc_model = roc_auc_score(ym_va, p_model)
            auroc_kmer = roc_auc_score(yk_va, p_kmer)
            res_mean, res_lo, res_hi = boot_residual_ci(ym_va, p_model, p_kmer)
            sig = (res_lo > 0)  # model significantly beats k-mer

            row = dict(
                ckpt=ck, task=task,
                model_auroc=round(auroc_model, 4),
                kmer_auroc=round(auroc_kmer, 4),
                residual=round(auroc_model - auroc_kmer, 4),
                resid_ci_lo=round(res_lo, 4), resid_ci_hi=round(res_hi, 4),
                model_beats_kmer=bool(sig),
                C_model=C_m, C_kmer=C_k,
                n_tr=len(ym_tr), n_va=len(ym_va),
            )
            rows.append(row)
            flag = "✅ >kmer" if sig else "≈/<kmer"
            print(f"  {ck:38s} {task:16s} model={auroc_model:.4f} kmer={auroc_kmer:.4f} "
                  f"resid={auroc_model-auroc_kmer:+.4f} [{res_lo:+.4f},{res_hi:+.4f}] {flag}")

    # Per-checkpoint mean across tasks
    df = pd.DataFrame(rows)
    if len(df):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"\nSaved: {a.out}")
        print("\n=== Mean across tasks (model vs k-mer) ===")
        g = df.groupby('ckpt').agg(
            model=('model_auroc', 'mean'),
            kmer=('kmer_auroc', 'mean'),
            residual=('residual', 'mean'),
        ).round(4)
        print(g.to_string())


if __name__ == "__main__":
    main()
