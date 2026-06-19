"""ViroBench leave-family-out (LOFO) host probe — phylogeny-disjoint stress test.

Why this exists (read me):
  The host/times split still shares ~97% of families train↔test, so composition
  can transfer. This probe removes that: it predicts the host of an ENTIRE viral
  family that was NEVER seen during probe training. Held-out families differ in
  composition from the training families, so a k-mer model cannot simply memorise
  "this composition → this host" and transfer it. If a model's embedding beats the
  matched k-mer baseline on truly unseen families, that is composition-immune host
  capability; if both fail equally (residual ≈ 0) the test is inconclusive (a null
  here is NOT proof of absence — this axis is intentionally a robustness check, not
  the headline, because ViroBench is 84% Orthomyxoviridae and non-influenza
  families are small).

Design (leave-one-family-out CV, matched k-mer control):
  * Reuse the EXACT cached host/times embeddings (no new GPU run). Family labels
    are joined back onto the cached subsample by taxid.
  * Eligible test families: ≥ MIN_FAM rows and ≥ 2 host classes present.
  * For each eligible family f: train probe on ALL OTHER rows, predict host_label
    on f. Pool out-of-fold predictions across all eligible families.
  * Model macro-AUROC vs k-mer macro-AUROC on the pooled OOF set; residual +
    paired bootstrap 95% CI. Locked checkpoints already used the f64-fused path
    when their embeddings were extracted.

Usage:
  python scripts/virobench_lofo_probe.py \
      --ckpts pretrained unlocked_ft M_a300k locked_no_ft \
      --out experiments/exp3_virobench/host_lofo_probe.csv
"""
from __future__ import annotations
import os, sys, csv, argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.virobench_host_probe import (
    VIROBENCH_SNAP, SPLIT_DIR, HOST_CLASSES, EMB_DIR, KMER_K, SEED, N_BOOT,
    kmer_frequencies, fit_probe, align_proba, macro_auroc,
)

MIN_FAM = 40            # min rows for a family to be an eligible held-out fold
MIN_HOSTS = 2           # min distinct host classes in a held-out family


def load_family_table() -> pd.DataFrame:
    """Cached subsample (row order == embedding order) with family joined by taxid."""
    sd = pd.read_parquet(f"{EMB_DIR}/shared_data.parquet")
    # IMPORTANT: keep file order = [train rows ..., test rows ...] to match npz.
    meta = pd.concat([
        pd.read_csv(f"{VIROBENCH_SNAP}/{SPLIT_DIR}/train.csv"),
        pd.read_csv(f"{VIROBENCH_SNAP}/{SPLIT_DIR}/test.csv"),
    ])[["taxid", "family"]].drop_duplicates("taxid")
    m = sd.merge(meta, on="taxid", how="left")
    assert (m["_split"].values[:int((m._split == "train").sum())] == "train").all(), \
        "row order not [train..., test...]"
    return m


def load_embeddings(ckpt_name: str) -> np.ndarray | None:
    npz = f"{EMB_DIR}/{ckpt_name}.npz"
    if not os.path.exists(npz):
        return None
    d = np.load(npz, allow_pickle=True)
    return np.concatenate([d["X_train"], d["X_test"]], axis=0), bool(d["fused"])


def eligible_families(df: pd.DataFrame) -> list[str]:
    g = df.groupby("family").agg(n=("taxid", "size"), hosts=("host_label", "nunique"))
    elig = g[(g.n >= MIN_FAM) & (g.hosts >= MIN_HOSTS)]
    return sorted(elig.index.tolist())


def lofo_oof_proba(X: np.ndarray, y: np.ndarray, fam: np.ndarray,
                   fams: list[str], n_classes: int):
    """Leave-one-family-out: train on others, predict held-out family. Pool OOF."""
    test_mask = np.isin(fam, fams)
    oof = np.full((len(y), n_classes), np.nan, dtype=float)
    for f in fams:
        te = (fam == f)
        tr = ~te                      # train on ALL other rows (incl. other fams)
        proba, _, cls = fit_probe(X[tr], y[tr], X[te])
        oof[te] = align_proba(proba, cls, n_classes)
    return oof, test_mask


def boot_residual_ci(y, model_oof, kmer_oof, classes, mask, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    idx_all = np.where(mask)[0]
    yt = y[idx_all]
    mp = model_oof[idx_all]
    kp = kmer_oof[idx_all]
    res = []
    n = len(idx_all)
    for _ in range(n_boot):
        bi = rng.integers(0, n, n)
        m = macro_auroc(yt[bi], mp[bi], classes)
        k = macro_auroc(yt[bi], kp[bi], classes)
        if not (np.isnan(m) or np.isnan(k)):
            res.append(m - k)
    res = np.array(res)
    return float(np.mean(res)), float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+',
                    default=["pretrained", "locked_no_ft", "unlocked_ft", "M_a300k"])
    ap.add_argument('--kmer_k', type=int, default=KMER_K)
    ap.add_argument('--out', default='experiments/exp3_virobench/host_lofo_probe.csv')
    a = ap.parse_args()

    df = load_family_table()
    fams = eligible_families(df)
    nC = len(HOST_CLASSES)
    y = df["y"].values
    fam = df["family"].values
    seqs = df["sequence"].tolist()
    print(f"[lofo] eligible held-out families ({len(fams)}): {fams}")
    print(f"[lofo] pooled OOF rows = {int(np.isin(fam, fams).sum())} of {len(df)}")

    # k-mer baseline (same LOFO folds, same rows)
    Xk = kmer_frequencies(seqs, a.kmer_k)
    kmer_oof, mask = lofo_oof_proba(Xk, y, fam, fams, nC)
    classes = range(nC)
    kmer_auc = macro_auroc(y[mask], kmer_oof[mask], classes)
    print(f"[kmer] LOFO macro-AUROC = {kmer_auc:.4f}")

    rows = []
    for name in a.ckpts:
        emb = load_embeddings(name)
        if emb is None:
            print(f"[warn] no embeddings for {name}; skipping"); continue
        X, fused = emb
        model_oof, _ = lofo_oof_proba(X, y, fam, fams, nC)
        m_auc = macro_auroc(y[mask], model_oof[mask], classes)
        rmean, rlo, rhi = boot_residual_ci(y, model_oof, kmer_oof, classes, mask)
        sig = "yes" if rlo > 0 else ("neg" if rhi < 0 else "ns")
        rows.append(dict(ckpt=name, fused=fused, n_oof=int(mask.sum()),
                         n_families=len(fams), model_auroc=round(m_auc, 4),
                         kmer_auroc=round(kmer_auc, 4), residual=round(rmean, 4),
                         ci_lo=round(rlo, 4), ci_hi=round(rhi, 4), beats_kmer=sig,
                         kmer_k=a.kmer_k))
        print(f"[probe] {name:13s} model={m_auc:.4f} kmer={kmer_auc:.4f} "
              f"residual={rmean:+.4f} [{rlo:+.4f},{rhi:+.4f}] beats_kmer={sig}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[done] wrote {a.out}")


if __name__ == '__main__':
    main()
