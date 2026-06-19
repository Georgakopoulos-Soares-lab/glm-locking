"""5-fold CV probe — no held-out test split.

Rationale
---------
Evo was pretrained on all publicly available viral/bacterial genomes. Neither
HVUE nor ViroBench held any sequences back from that pretraining corpus, so
the benchmark's train/test/validation split boundary is NOT a "truly unseen to
the model" boundary. The temporal ViroBench split (≤2017 / ≥2020) controls for
recency of emergence, not for model exposure.

A fixed held-out test split is therefore potentially misleading: a model that
memorised many test-set genomes during pretraining gains unwarranted credit.
This script addresses that by running 5-fold stratified CV on the FULL combined
dataset (train ∪ test/validation) for both HVUE (binary) and ViroBench host
(multiclass). The k-mer baseline gets the identical CV treatment. Residual and
bootstrap CIs are computed on the pooled out-of-fold predictions.

No new GPU extraction needed: existing cached embeddings are concatenated.

Fixed C (validated from existing rigorous analysis — consistent across all tasks/ckpts):
  model=10.0, k-mer=0.01

Usage:
  CUDA_VISIBLE_DEVICES="" python scripts/cv_probe.py \
      --ckpts pretrained unlocked_ft \
      --out_hvue results/hvue_cv_probe.csv \
      --out_virobench experiments/exp3_virobench/host_cv_probe.csv
"""
from __future__ import annotations
import os, sys, csv, argparse, itertools
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────
HVUE_DIR      = "data/hvue"
HVUE_EMB_DIR  = "results/hvue_embeddings"
VB_EMB_DIR    = "experiments/exp3_virobench/host_times_embeddings"

HVUE_TASKS   = ["Host_Tropism", "Pathogenecity", "Transmissibility"]
HOST_CLASSES = ["A", "B", "C", "D1", "D2", "D3", "E", "F"]
NUCLEOTIDES  = ['A', 'C', 'G', 'T']

N_HVUE_TRAIN = 3000   # must match hvue_extract_one_ckpt.py
N_HVUE_VAL   = 2000
MAX_LEN      = 4096

SEED  = 0
N_CV  = 5
N_BOOT = 5000
KMER_K = 4

# Validated from existing rigorous analysis (C identical across all tasks/ckpts)
C_MODEL = 10.0
C_KMER  = 0.01

# HVUE embedding naming: our canonical key → prefix used when extracting
HVUE_CKPT_ALIAS = {
    "pretrained":   "pretrained",
    "unlocked_ft":  "ft_unlocked_25k_v2_unlocked",
}

# ViroBench embedding naming matches our canonical keys directly
VB_CKPTS_AVAILABLE = ["pretrained", "locked_no_ft", "unlocked_ft", "M_a300k"]


# ── k-mer features ────────────────────────────────────────────────────────────

def kmer_frequencies(seqs: list[str], k: int) -> np.ndarray:
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


# ── CV probe ──────────────────────────────────────────────────────────────────

def cv_oof_proba(X: np.ndarray, y: np.ndarray, C: float,
                 n_classes: int) -> np.ndarray:
    """5-fold stratified CV → pooled OOF probability matrix (n × n_classes)."""
    oof = np.zeros((len(y), n_classes), dtype=float)
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=3000, random_state=SEED,
                                 multi_class='ovr' if n_classes > 2 else 'auto')
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))
        for j, c in enumerate(clf.classes_):
            oof[te, int(c)] = p[:, j]
    return oof


def binary_auroc(y: np.ndarray, oof: np.ndarray) -> float:
    return float(roc_auc_score(y, oof[:, 1]))


def macro_auroc(y: np.ndarray, oof: np.ndarray, n_classes: int) -> float:
    aucs = []
    for ci in range(n_classes):
        yt = (y == ci).astype(int)
        if 0 < yt.sum() < len(yt):
            aucs.append(roc_auc_score(yt, oof[:, ci]))
    return float(np.mean(aucs)) if aucs else float('nan')


def boot_residual(y: np.ndarray, model_oof: np.ndarray, kmer_oof: np.ndarray,
                  binary: bool, n_classes: int):
    """Paired bootstrap: distribution of (AUROC_model − AUROC_kmer) on OOF preds."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    res = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        try:
            if binary:
                m = roc_auc_score(yb, model_oof[idx, 1])
                k = roc_auc_score(yb, kmer_oof[idx, 1])
            else:
                m = macro_auroc(yb, model_oof[idx], n_classes)
                k = macro_auroc(yb, kmer_oof[idx], n_classes)
            if not (np.isnan(m) or np.isnan(k)):
                res.append(m - k)
        except Exception:
            pass
    res = np.array(res)
    return float(np.mean(res)), float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


# ── Reproduce exact embedding-extraction sampling ────────────────────────────

def repro_sample(task: str, split: str, n: int) -> pd.DataFrame:
    """Reproduce the EXACT rows used during HVUE embedding extraction (must match)."""
    df = pd.read_parquet(f"{HVUE_DIR}/{task}_{split}.parquet")
    per_class = n // df['label'].nunique()
    sub = df.groupby('label', group_keys=False).apply(
        lambda g: g.sample(min(per_class, len(g)), random_state=0),
        include_groups=False,
    )
    sub = df.loc[sub.index]
    sub = sub.sample(frac=1, random_state=0)
    return sub


# ── HVUE ─────────────────────────────────────────────────────────────────────

def run_hvue(ckpts: list[str], kmer_k: int, out_path: str) -> list[dict]:
    print("=" * 65)
    print("HVUE — 5-fold CV on combined train+validation (no holdout)")
    print(f"  Rationale: Evo pretrained on same corpus; split boundary is not")
    print(f"  'model-unseen'. C_model={C_MODEL}, C_kmer={C_KMER} (pre-validated).")
    print("=" * 65)

    rows = []
    for task in HVUE_TASKS:
        # Reproduce exact extraction sample (matches npz row order)
        tr_df = repro_sample(task, 'train',      N_HVUE_TRAIN)
        va_df = repro_sample(task, 'validation',  N_HVUE_VAL)
        seqs = (tr_df['sequence'].str.slice(0, MAX_LEN).tolist() +
                va_df['sequence'].str.slice(0, MAX_LEN).tolist())
        y = np.concatenate([tr_df['label'].values.astype(int),
                            va_df['label'].values.astype(int)])
        n = len(y)
        print(f"\n[{task}]  n={n}  pos_rate={y.mean():.3f}")

        # k-mer baseline (same rows, same CV folds)
        Xk = kmer_frequencies(seqs, kmer_k)
        print(f"  computing k-mer CV (k={kmer_k}, C={C_KMER})...", flush=True)
        kmer_oof = cv_oof_proba(Xk, y, C_KMER, n_classes=2)
        kmer_auc = binary_auroc(y, kmer_oof)
        print(f"  k-mer AUROC (CV OOF) = {kmer_auc:.4f}")

        for ckpt in ckpts:
            alias = HVUE_CKPT_ALIAS.get(ckpt)
            if alias is None:
                print(f"  SKIP {ckpt}: no HVUE alias defined"); continue
            tr_npz = f"{HVUE_EMB_DIR}/{alias}_{task}_train.npz"
            va_npz = f"{HVUE_EMB_DIR}/{alias}_{task}_validation.npz"
            if not (os.path.exists(tr_npz) and os.path.exists(va_npz)):
                print(f"  SKIP {ckpt} ({alias}): embeddings not found"); continue

            dtr = np.load(tr_npz, allow_pickle=True)
            dva = np.load(va_npz, allow_pickle=True)

            # Sanity: embedding labels must match the repro sample (same rows)
            assert np.array_equal(dtr['y'].astype(int), tr_df['label'].values.astype(int)), \
                f"train label mismatch {ckpt}/{task}"
            assert np.array_equal(dva['y'].astype(int), va_df['label'].values.astype(int)), \
                f"validation label mismatch {ckpt}/{task}"

            X = np.concatenate([dtr['X'], dva['X']], axis=0)
            print(f"  [{ckpt}] computing model CV (C={C_MODEL}, dim={X.shape[1]})...", flush=True)
            model_oof = cv_oof_proba(X, y, C_MODEL, n_classes=2)
            m_auc = binary_auroc(y, model_oof)
            rmean, rlo, rhi = boot_residual(y, model_oof, kmer_oof, binary=True, n_classes=2)
            sig = "yes" if rlo > 0 else ("neg" if rhi < 0 else "ns")
            rows.append(dict(dataset='HVUE', task=task, ckpt=ckpt, n=n,
                             model_auroc=round(m_auc, 4), kmer_auroc=round(kmer_auc, 4),
                             residual=round(rmean, 4), ci_lo=round(rlo, 4), ci_hi=round(rhi, 4),
                             beats_kmer=sig, kmer_k=kmer_k, C_model=C_MODEL, C_kmer=C_KMER))
            print(f"  {ckpt:25s} model={m_auc:.4f} kmer={kmer_auc:.4f} "
                  f"residual={rmean:+.4f} [{rlo:+.4f},{rhi:+.4f}] beats_kmer={sig}")

    if not rows:
        print("[warn] no HVUE rows produced"); return []

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[done] HVUE CV → {out_path}")
    return rows


# ── ViroBench host ────────────────────────────────────────────────────────────

def run_virobench(ckpts: list[str], kmer_k: int, out_path: str) -> list[dict]:
    print("\n" + "=" * 65)
    print("ViroBench host — 5-fold CV on combined train+test (no holdout)")
    print(f"  C_model={C_MODEL}, C_kmer={C_KMER} (pre-validated).")
    print("=" * 65)

    # Shared data (both train and test rows, order matches npz)
    sd = pd.read_parquet(f"{VB_EMB_DIR}/shared_data.parquet")
    seqs = sd['sequence'].str.slice(0, MAX_LEN).tolist()
    y = sd['y'].values.astype(int)
    n = len(y)
    nC = len(HOST_CLASSES)
    print(f"  n={n}  classes={nC}")

    # k-mer baseline (same rows, same CV folds)
    Xk = kmer_frequencies(seqs, kmer_k)
    print(f"  computing k-mer CV (k={kmer_k}, C={C_KMER})...", flush=True)
    kmer_oof = cv_oof_proba(Xk, y, C_KMER, n_classes=nC)
    kmer_auc = macro_auroc(y, kmer_oof, nC)
    print(f"  k-mer macro-AUROC (CV OOF) = {kmer_auc:.4f}")

    rows = []
    for ckpt in ckpts:
        npz = f"{VB_EMB_DIR}/{ckpt}.npz"
        if not os.path.exists(npz):
            print(f"  SKIP {ckpt}: embeddings not found"); continue
        d = np.load(npz, allow_pickle=True)
        X = np.concatenate([d['X_train'], d['X_test']], axis=0)
        if len(X) != n:
            print(f"  SKIP {ckpt}: embedding rows {len(X)} != data rows {n}"); continue
        fused = bool(d.get('fused', False))
        print(f"  [{ckpt}] computing model CV (C={C_MODEL}, dim={X.shape[1]})...", flush=True)
        model_oof = cv_oof_proba(X, y, C_MODEL, n_classes=nC)
        m_auc = macro_auroc(y, model_oof, nC)
        rmean, rlo, rhi = boot_residual(y, model_oof, kmer_oof, binary=False, n_classes=nC)
        sig = "yes" if rlo > 0 else ("neg" if rhi < 0 else "ns")
        rows.append(dict(dataset='ViroBench_host', task='host', ckpt=ckpt,
                         fused=fused, n=n, n_classes=nC,
                         model_auroc=round(m_auc, 4), kmer_auroc=round(kmer_auc, 4),
                         residual=round(rmean, 4), ci_lo=round(rlo, 4), ci_hi=round(rhi, 4),
                         beats_kmer=sig, kmer_k=kmer_k, C_model=C_MODEL, C_kmer=C_KMER))
        print(f"  {ckpt:25s} model={m_auc:.4f} kmer={kmer_auc:.4f} "
              f"residual={rmean:+.4f} [{rlo:+.4f},{rhi:+.4f}] beats_kmer={sig}")

    if not rows:
        print("[warn] no ViroBench rows produced"); return []

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[done] ViroBench host CV → {out_path}")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', default=['pretrained', 'unlocked_ft'])
    ap.add_argument('--kmer_k', type=int, default=KMER_K)
    ap.add_argument('--out_hvue',      default='results/hvue_cv_probe.csv')
    ap.add_argument('--out_virobench', default='experiments/exp3_virobench/host_cv_probe.csv')
    ap.add_argument('--skip_hvue',      action='store_true')
    ap.add_argument('--skip_virobench', action='store_true')
    a = ap.parse_args()

    if not a.skip_hvue:
        run_hvue(a.ckpts, a.kmer_k, a.out_hvue)
    if not a.skip_virobench:
        run_virobench(a.ckpts, a.kmer_k, a.out_virobench)


if __name__ == '__main__':
    main()
