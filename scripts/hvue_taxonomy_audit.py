"""Priority 2: HVUE taxonomy confound audit via k-mer composition baseline.

The HVUE parquet files (sequence + label only, no taxid) prevent literal taxonomy-only
baselines. Instead: sequence-composition baselines using k-mer frequencies. k-mer
composition captures taxonomic signal (different viral families have distinct genomic
signatures). If a k-mer probe AUROC ≈ model AUROC, HVUE labels are largely predictable
from sequence composition → taxonomic confound.

Also reports: GC content, sequence length, and dinucleotide frequency baselines.
"""
from __future__ import annotations
import os, sys, csv, itertools, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

HVUE_DIR = "data/hvue"
TASKS = ["Host_Tropism", "Pathogenecity", "Transmissibility"]
SPLITS = ["train", "validation", "test"]
MODEL_RESULTS_CSV = "results/hvue_probe.csv"
OUT_CSV = "results/hvue_composition_audit.csv"
MAX_SAMPLES = 5000  # subsample for speed (k-mer features are wide)


# ── Sequence composition features ───────────────────────────────────────────

NUCLEOTIDES = ['A', 'C', 'G', 'T']

def kmer_frequencies(seqs, k):
    """Compute k-mer frequency matrix [n_seqs, 4^k]."""
    kmers = [''.join(p) for p in itertools.product(NUCLEOTIDES, repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)), dtype=np.float32)
    for i, s in enumerate(seqs):
        s_clean = s.upper()
        for j in range(len(s_clean) - k + 1):
            km = s_clean[j:j+k]
            if km in kmer_to_idx:
                X[i, kmer_to_idx[km]] += 1
        total = X[i].sum()
        if total > 0:
            X[i] /= total
    return X


def gc_content(seqs):
    """GC content per sequence."""
    gc = np.zeros((len(seqs), 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        s_upper = s.upper()
        g = s_upper.count('G')
        c = s_upper.count('C')
        gc[i, 0] = (g + c) / max(len(s_upper), 1)
    return gc


def dinucleotide_freqs(seqs):
    """Dinucleotide frequencies (16-dim)."""
    return kmer_frequencies(seqs, k=2)


def sequence_length(seqs):
    """Log sequence length."""
    return np.log10([max(len(s), 1) for s in seqs]).reshape(-1, 1)


# ── Probe training ──────────────────────────────────────────────────────────

def probe_binary(X_tr, y_tr, X_te, y_te, C_values=[0.01, 0.1, 1.0, 10.0], seed=0):
    """Train logistic regression, return metrics dict."""
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_te_s = sc.transform(X_te)

    best = None
    best_clf = None
    for C in C_values:
        clf = LogisticRegression(max_iter=2000, C=C, random_state=seed, n_jobs=-1)
        clf.fit(X_tr_s, y_tr)
        p = clf.predict_proba(X_te_s)[:, 1]
        try:
            auroc = roc_auc_score(y_te, p)
        except Exception:
            auroc = float('nan')
        if best is None or auroc > best[0]:
            best = (auroc, C)
            best_clf = clf

    auroc, C = best
    clf = best_clf
    pred = clf.predict(X_te_s)
    return dict(
        auroc=float(auroc),
        acc=float(accuracy_score(y_te, pred)),
        f1=float(f1_score(y_te, pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_te, pred)),
        best_C=float(C),
        n_train=int(len(y_tr)),
        n_test=int(len(y_te)),
        label_balance_te=float(y_te.mean()),
        n_features=X_tr.shape[1],
    )


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HVUE Composition Confound Audit")
    print("k-mer frequency baselines vs model embedding probes")
    print("=" * 70)

    # 1. Load HVUE data
    print("\n1. Loading HVUE parquet data...")
    data = {}
    for task in TASKS:
        data[task] = {}
        for split in SPLITS:
            path = os.path.join(HVUE_DIR, f"{task}_{split}.parquet")
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path)
            if len(df) > MAX_SAMPLES:
                df = df.sample(MAX_SAMPLES, random_state=42)
            data[task][split] = df
            pos_frac = df['label'].mean()
            print(f"  {task}/{split}: {len(df)} seqs, label_balance={pos_frac:.3f}")

    # 2. Load model AUROC for comparison
    model_auroc = {}
    if os.path.exists(MODEL_RESULTS_CSV):
        with open(MODEL_RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                ckpt = row.get('ckpt', '')
                task_name = row.get('task', '')
                auroc = float(row.get('auroc', 0))
                key = (ckpt, task_name)
                if key not in model_auroc or auroc > model_auroc[key]:
                    model_auroc[key] = auroc
        print(f"\n  Loaded {len(model_auroc)} model results from {MODEL_RESULTS_CSV}")

    # 3. Compute composition baselines
    print("\n2. Computing composition baselines...")
    all_results = []

    BASELINES = {
        'kmer3': lambda seqs: kmer_frequencies(seqs, k=3),   # 64-dim
        'kmer4': lambda seqs: kmer_frequencies(seqs, k=4),   # 256-dim
        'dinuc': lambda seqs: dinucleotide_freqs(seqs),       # 16-dim
        'gc_content': lambda seqs: gc_content(seqs),          # 1-dim
        'length': lambda seqs: sequence_length(seqs),          # 1-dim
        'gc+length': lambda seqs: np.hstack([gc_content(seqs), sequence_length(seqs)]),  # 2-dim
    }

    for task in TASKS:
        if task not in data or 'train' not in data[task] or 'test' not in data[task]:
            print(f"  {task}: missing splits, skipping")
            continue

        df_tr = data[task]['train']
        df_te = data[task]['test']
        seqs_tr = df_tr['sequence'].tolist()
        y_tr = df_tr['label'].values.astype(int)
        seqs_te = df_te['sequence'].tolist()
        y_te = df_te['label'].values.astype(int)

        print(f"\n  {task} (n_tr={len(seqs_tr)}, n_te={len(seqs_te)}, balance_tr={y_tr.mean():.3f})")

        for name, feature_fn in BASELINES.items():
            t0 = time.time()
            try:
                if name == 'kmer4' and len(seqs_tr) > 2000:
                    idx = np.random.choice(len(seqs_tr), min(2000, len(seqs_tr)), replace=False)
                    X_tr = feature_fn([seqs_tr[i] for i in idx])
                    y_tr_sub = y_tr[idx]
                    X_te = feature_fn(seqs_te)
                else:
                    X_tr = feature_fn(seqs_tr)
                    y_tr_sub = y_tr
                    X_te = feature_fn(seqs_te)

                r = probe_binary(X_tr, y_tr_sub, X_te, y_te)
                r['task'] = task
                r['baseline'] = name
                r['time_s'] = round(time.time() - t0, 1)
                all_results.append(r)
                print(f"    {name:12s}  AUROC={r['auroc']:.4f}  Acc={r['acc']:.4f}  F1={r['f1']:.4f}  (n_feat={r['n_features']}, {r['time_s']:.1f}s)")
            except Exception as e:
                print(f"    {name:12s}  FAILED: {e}")

    # 4. Compare with model AUROC
    print("\n3. Comparison: model embeddings vs best composition baseline")
    print("-" * 75)
    print(f"{'Task':25s} {'Best k-mer':12s} {'Model AUROC':14s} {'Delta':10s} {'Verdict'}")
    print("-" * 75)

    for task in TASKS:
        task_results = [r for r in all_results if r['task'] == task]
        if not task_results:
            continue

        best_comp = max(task_results, key=lambda r: r['auroc'])
        best_auroc = best_comp['auroc']
        best_name = best_comp['baseline']

        # Find model AUROC for M on this task
        model_auroc_for_task = None
        for (ckpt, t), a in model_auroc.items():
            if t == task:
                if model_auroc_for_task is None or a > model_auroc_for_task:
                    model_auroc_for_task = a

        if model_auroc_for_task is not None:
            delta = model_auroc_for_task - best_auroc
            if delta > 0.10:
                verdict = "MODEL >> composition (good — HVUE measures more than taxonomy)"
            elif delta > 0.03:
                verdict = "Model > composition (modest signal beyond composition)"
            elif abs(delta) <= 0.03:
                verdict = "≈ TIED — POSSIBLE CONFOUND (label predictable from composition)"
            else:
                verdict = "COMPOSITION WINS (label is compositional, not semantic)"
            print(f"{task:25s} {best_name}={best_auroc:.4f}   {model_auroc_for_task:.4f}         {delta:+.4f}     {verdict}")
        else:
            print(f"{task:25s} {best_name}={best_auroc:.4f}   (no model data)")

    # 5. Save
    if all_results:
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        with open(OUT_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_results:
                for k in keys:
                    if k not in r:
                        r[k] = ''
            w.writerows(all_results)
        print(f"\nSaved: {OUT_CSV}")

    # 6. Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
- k-mer composition captures taxonomic signal (different viral families have
  distinct genomic signatures). It is NOT a measure of functional understanding.
- If k-mer AUROC ≈ model AUROC: the HVUE label may be predictable from taxonomy
  (genomic composition) alone. This doesn't mean the model's AUROC is invalid,
  but it means we cannot rule out taxonomic confounding.
- If model AUROC >> k-mer AUROC: the model's embeddings carry additional signal
  beyond composition → stronger evidence of capability measurement.
- This audit is reported as a LIMITATION regardless of outcome, because we
  cannot do a literal taxonomy-only baseline without NCBI taxids.
""")


if __name__ == "__main__":
    main()
