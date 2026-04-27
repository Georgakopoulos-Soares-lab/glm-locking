"""Bootstrap CI + DeLong-style paired test on HVUE probe AUROCs.

Compares each (ckpt, task) pair against pretrained baseline using the *same*
val embeddings (so probe weights differ only via training subsample → we
bootstrap the val set, retraining probe each iteration is too slow,
so we use a fixed probe and bootstrap the AUROC over val examples — that
is the standard DeLong-equivalent comparison).
"""
import argparse, glob, os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def load(emb_dir, ckpt, task, split):
    p = os.path.join(emb_dir, f"{ckpt}_{task}_{split}.npz")
    if not os.path.exists(p): return None
    d = np.load(p)
    return d["X"], d["y"]

def fit_probe(Xtr, ytr, Cs=(0.01, 0.1, 1.0, 10.0)):
    sc = StandardScaler().fit(Xtr)
    Xtrs = sc.transform(Xtr)
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(Xtrs, ytr)
        s = clf.decision_function(Xtrs)
        a = roc_auc_score(ytr, s)
        if best is None or a > best[0]:
            best = (a, clf, sc, C)
    return best[1], best[2]

def bootstrap_auroc_diff(scores_a, scores_b, y, n_boot=2000, seed=0):
    """For each bootstrap of val examples: AUROC(a) − AUROC(b)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            a = roc_auc_score(y[idx], scores_a[idx])
            b = roc_auc_score(y[idx], scores_b[idx])
            diffs.append(a - b)
        except ValueError:
            continue
    diffs = np.array(diffs)
    return diffs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="results/hvue_embeddings")
    ap.add_argument("--baseline", default="pretrained")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    tasks = ["Host_Tropism", "Pathogenecity", "Transmissibility"]
    ckpts = sorted({os.path.basename(p).split("_Host_Tropism_")[0]
                    for p in glob.glob(os.path.join(args.emb_dir, "*_Host_Tropism_train.npz"))})
    print(f"ckpts: {ckpts}")

    rows = []
    for task in tasks:
        # Baseline probe (fit on baseline train, score baseline val)
        b = load(args.emb_dir, args.baseline, task, "train")
        bv = load(args.emb_dir, args.baseline, task, "validation")
        if b is None or bv is None:
            continue
        clf_b, sc_b = fit_probe(*b)
        scores_b = clf_b.decision_function(sc_b.transform(bv[0]))
        y_v = bv[1]
        auroc_b = roc_auc_score(y_v, scores_b)
        print(f"\n=== {task} ===   baseline ({args.baseline}) AUROC = {auroc_b:.4f}")

        for ckpt in ckpts:
            if ckpt == args.baseline:
                continue
            tr = load(args.emb_dir, ckpt, task, "train")
            va = load(args.emb_dir, ckpt, task, "validation")
            if tr is None or va is None:
                continue
            # Each ckpt has its OWN embeddings, so val labels need to match.
            # Sort labels match? We stratified-sample with same seed assumption — verify by label order
            # For paired bootstrap, re-evaluate baseline scores on this ckpt's val labels via
            # baseline probe on ckpt val embeddings? That's WRONG because baseline probe was
            # fit on baseline embeddings. The standard approach: paired test only makes sense
            # if the val examples (sequences) are the same. They ARE — we sampled deterministically.
            assert np.array_equal(va[1], y_v), f"label order mismatch for {ckpt} {task}"
            clf_c, sc_c = fit_probe(*tr)
            scores_c = clf_c.decision_function(sc_c.transform(va[0]))
            auroc_c = roc_auc_score(y_v, scores_c)
            diffs = bootstrap_auroc_diff(scores_c, scores_b, y_v, n_boot=args.n_boot)
            ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
            p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
            sign = "+" if (auroc_c - auroc_b) > 0 else ""
            stars = ""
            if ci_lo > 0 or ci_hi < 0: stars = "*"
            if p_two < 0.01: stars = "**"
            if p_two < 0.001: stars = "***"
            print(f"  {ckpt:30s} AUROC={auroc_c:.4f}  Δ={sign}{auroc_c-auroc_b:+.4f}  "
                  f"95%CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  p={p_two:.4f} {stars}")
            rows.append(dict(task=task, ckpt=ckpt, auroc=auroc_c, baseline_auroc=auroc_b,
                             delta=auroc_c-auroc_b, ci_lo=ci_lo, ci_hi=ci_hi, p=p_two))

    import csv
    with open("results/hvue_significance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task","ckpt","auroc","baseline_auroc","delta","ci_lo","ci_hi","p"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print("\nwrote results/hvue_significance.csv")

if __name__ == "__main__":
    main()
