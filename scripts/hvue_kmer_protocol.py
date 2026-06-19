"""Model-free k-mer baseline on the HVUE benchmark, under the HViLM paper protocol.

Purpose: contextualise HViLM's headline 95%+ accuracies by asking the cleanest
possible question — how much of each HVUE task is solved by raw sequence
composition alone, with NO model? We train a plain k-mer-frequency logistic
regression on the paper's OWN train split and report on the paper's OWN test
split, using the paper's metrics (Accuracy / F1 / MCC). A majority-class baseline
is reported alongside as the trivial floor.

If k-mer LR approaches HViLM's accuracy (and ≈99.9% on the single-family Calici /
within-family splits), the headline reflects sequence composition rather than a
learned virological capability — fully model-free, independent of any checkpoint.

Datasets (duttaprat/HVUE, natural splits, 1000 bp chunks, columns: sequence,label):
  Pathogenicity:    CINI, BVBRC_CoV, BVBRC_Calci
  Host tropism:     Host_Tropism (VHDB)
  Transmissibility: Coronovirdae, Orthomyxovirdae, Calcivirdae

HViLM reported accuracies (for the side-by-side): CINI 87.74, BVBRC_CoV 98.26,
BVBRC_Calci 99.95, VHDB 96.25, Corona 97.45, Orthomyxo 95.62, Calici 99.95.

CPU-only. Usage:
  HF_HOME=/data/huggingface_cache python scripts/hvue_kmer_protocol.py \
      --kmer_k 4 --out experiments/hvue_kmer_protocol.csv
"""
from __future__ import annotations
import os, sys, csv, argparse, itertools
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

NUCLEOTIDES = ['A', 'C', 'G', 'T']
SEED = 0
C_GRID = [0.01, 0.1, 1.0, 10.0]
MAX_TRAIN = 60000   # stratified cap for tractability on the largest sets
MAX_TEST = 20000

# (display_name, hf_subdir, HViLM_reported_accuracy_pct)
DATASETS = [
    ("Patho/CINI",          "Pathogenecity/CINI",              87.74),
    ("Patho/BVBRC_CoV",     "Pathogenecity/BVBRC_CoV",         98.26),
    ("Patho/BVBRC_Calci",   "Pathogenecity/BVBRC_Calci",       99.95),
    ("Host/VHDB",           "Host_Tropism",                    96.25),
    ("Trans/Coronaviridae", "Transmissibility/Coronovirdae",   97.45),
    ("Trans/Orthomyxo",     "Transmissibility/Orthomyxovirdae",95.62),
    ("Trans/Caliciviridae", "Transmissibility/Calcivirdae",    99.95),
]


def load_csv(subdir, split):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("duttaprat/HVUE", f"{subdir}/{split}.csv", repo_type="dataset")
    return pd.read_csv(p)


def stratified_cap(df, n):
    if len(df) <= n:
        return df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    frac = n / len(df)
    sub = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=SEED))
    return sub.sample(frac=1, random_state=SEED).reset_index(drop=True)


def kmer_frequencies(seqs, k):
    kmers = [''.join(p) for p in itertools.product(NUCLEOTIDES, repeat=k)]
    idx = {km: i for i, km in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper()
        for j in range(len(s) - k + 1):
            km = s[j:j + k]
            if km in idx:
                X[i, idx[km]] += 1
        tot = X[i].sum()
        if tot > 0:
            X[i] /= tot
    return X


def fit_eval(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_C, best = None, -np.inf
    for C in C_GRID:
        s = cross_val_score(LogisticRegression(max_iter=2000, C=C, random_state=SEED),
                            Xtr_s, ytr, cv=cv, scoring='f1_macro', n_jobs=-1).mean()
        if s > best:
            best, best_C = s, C
    clf = LogisticRegression(max_iter=3000, C=best_C, random_state=SEED).fit(Xtr_s, ytr)
    pred = clf.predict(Xte_s)
    return dict(
        acc=accuracy_score(yte, pred) * 100,
        f1_macro=f1_score(yte, pred, average='macro') * 100,
        f1_pos=f1_score(yte, pred, pos_label=1, average='binary') * 100,
        mcc=matthews_corrcoef(yte, pred) * 100,
        C=best_C,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kmer_k', type=int, default=4)
    ap.add_argument('--out', default='experiments/hvue_kmer_protocol.csv')
    a = ap.parse_args()

    rows = []
    for name, subdir, hvilm_acc in DATASETS:
        print(f"\n=== {name} ({subdir}) ===")
        tr = stratified_cap(load_csv(subdir, "train"), MAX_TRAIN)
        te = stratified_cap(load_csv(subdir, "test"), MAX_TEST)
        ytr, yte = tr["label"].values, te["label"].values
        maj_label = Counter(ytr).most_common(1)[0][0]
        maj_acc = (yte == maj_label).mean() * 100
        print(f"  train n={len(tr)} ({dict(sorted(Counter(ytr).items()))}), "
              f"test n={len(te)} ({dict(sorted(Counter(yte).items()))}), maj_acc={maj_acc:.2f}")
        Xtr = kmer_frequencies(tr["sequence"].tolist(), a.kmer_k)
        Xte = kmer_frequencies(te["sequence"].tolist(), a.kmer_k)
        m = fit_eval(Xtr, ytr, Xte, yte)
        gap = m['acc'] - hvilm_acc
        print(f"  kmer{a.kmer_k}: acc={m['acc']:.2f}  f1_macro={m['f1_macro']:.2f}  "
              f"mcc={m['mcc']:.2f}  | HViLM acc={hvilm_acc:.2f}  (gap {gap:+.2f})")
        rows.append(dict(dataset=name, n_train=len(tr), n_test=len(te),
                         majority_acc=round(maj_acc, 2),
                         kmer_acc=round(m['acc'], 2), kmer_f1_macro=round(m['f1_macro'], 2),
                         kmer_f1_pos=round(m['f1_pos'], 2), kmer_mcc=round(m['mcc'], 2),
                         hvilm_acc=hvilm_acc, acc_gap=round(gap, 2),
                         kmer_k=a.kmer_k, C=m['C']))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[done] wrote {a.out}")
    print("\nSUMMARY (model-free k-mer vs HViLM accuracy):")
    for r in rows:
        print(f"  {r['dataset']:22s} kmer {r['kmer_acc']:6.2f}  HViLM {r['hvilm_acc']:6.2f}  "
              f"(maj {r['majority_acc']:6.2f})")


if __name__ == '__main__':
    main()
