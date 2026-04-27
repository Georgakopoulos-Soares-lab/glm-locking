"""Train logistic-regression probes on Evo embeddings for HVUE tasks.

For each (ckpt, task) pair, train probe on `train` embeddings, evaluate on
`validation` embeddings. Reports AUROC, accuracy, F1, MCC.

Usage:
  python scripts/hvue_probe.py --emb_dir results/hvue_embeddings --out results/hvue_probe_results.csv
"""
import os, sys, argparse, glob, json
import numpy as np
from collections import defaultdict


def load(path):
    d = np.load(path, allow_pickle=True)
    return d['X'], d['y'], str(d['ckpt']), str(d['task']), str(d['split'])


def probe_one(X_tr, y_tr, X_va, y_va, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr); X_va_s = sc.transform(X_va)
    # small grid over C
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(max_iter=2000, C=C, random_state=seed, n_jobs=-1)
        clf.fit(X_tr_s, y_tr)
        p = clf.predict_proba(X_va_s)[:, 1]
        try:
            auroc = roc_auc_score(y_va, p)
        except Exception:
            auroc = float('nan')
        if best is None or auroc > best[0]:
            best = (auroc, C, clf, p)
    auroc, C, clf, p = best
    pred = (p > 0.5).astype(int)
    return dict(
        auroc=float(auroc),
        acc=float(accuracy_score(y_va, pred)),
        f1=float(f1_score(y_va, pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_va, pred)),
        n_train=int(len(y_tr)),
        n_val=int(len(y_va)),
        best_C=float(C),
        label_balance_val=float(y_va.mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb_dir', default='results/hvue_embeddings')
    ap.add_argument('--out', default='results/hvue_probe_results.csv')
    ap.add_argument('--out_json', default='results/hvue_probe_results.json')
    a = ap.parse_args()

    files = sorted(glob.glob(f'{a.emb_dir}/*.npz'))
    by_pair = defaultdict(dict)  # (ckpt, task) -> {'train':..., 'validation':...}
    for f in files:
        X, y, ck, task, split = load(f)
        by_pair[(ck, task)][split] = (X, y)

    rows = []
    for (ck, task), d in sorted(by_pair.items()):
        if 'train' not in d or 'validation' not in d:
            print(f'  skip {ck}/{task}: missing splits {list(d.keys())}'); continue
        X_tr, y_tr = d['train']; X_va, y_va = d['validation']
        # require at least 2 classes
        if len(set(y_tr)) < 2 or len(set(y_va)) < 2:
            print(f'  skip {ck}/{task}: single class'); continue
        r = probe_one(X_tr, y_tr, X_va, y_va)
        r.update(ckpt=ck, task=task)
        print(f'{ck:30s} {task:18s} AUROC={r["auroc"]:.3f} acc={r["acc"]:.3f} f1={r["f1"]:.3f} mcc={r["mcc"]:+.3f}  (n_tr={r["n_train"]} n_va={r["n_val"]})')
        rows.append(r)

    # CSV
    if rows:
        keys = ['ckpt','task','auroc','acc','f1','mcc','n_train','n_val','best_C','label_balance_val']
        with open(a.out, 'w') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r[k]) for k in keys) + '\n')
        with open(a.out_json, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f'\nwrote {a.out} and {a.out_json}  ({len(rows)} probes)')

        # Summary table
        print('\n=== Summary: AUROC by (ckpt, task) ===')
        ckpts = sorted({r['ckpt'] for r in rows})
        tasks = sorted({r['task'] for r in rows})
        print(f'{"ckpt":30s} ' + ' '.join(f'{t[:14]:>14s}' for t in tasks))
        for c in ckpts:
            line = f'{c:30s} '
            for t in tasks:
                v = next((r['auroc'] for r in rows if r['ckpt']==c and r['task']==t), None)
                line += f'{v:>14.3f}' if v is not None else f'{"-":>14s}'
                line += ' '
            print(line)


if __name__ == '__main__':
    main()
