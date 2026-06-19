"""Merge per-(ckpt,config) shard CSVs and compute paired-bootstrap CIs.

Produces:
  results/hvue_lora_fair_headtohead.csv   (matched + force_locked)
  results/hvue_lora_svd.csv               (SVD-chain matched runs)

Pairwise comparisons (per task, all on the best-LR shard):
  unlocked_ft  − pretrained     (FT uplift)
  M_a300k      − pretrained     (locked-attacker boost)
  locked_no_ft − pretrained     (clean lock test, no FT)
  M_a300k      − unlocked_ft    (lock marginal, CONFOUNDED — flagged)

Uses the saved per-sample .npz prediction files in results/hvue_lora_preds/
for paired bootstrap (10k resamples, 95% CI).
"""
from __future__ import annotations
import os, glob, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARDS_DIR = f"{REPO}/results/hvue_lora_shards"
PREDS_DIR  = f"{REPO}/results/hvue_lora_preds"

# Map ckpt → config to determine which preds key to look up
def preds_path(task, ckpt, lr, cfg):
    cfg_tag = "" if cfg == "full" else f"__{cfg}"
    return os.path.join(PREDS_DIR, f"{task}_{ckpt}_{lr:.0e}{cfg_tag}.npz")


def mcc_from_logits(logits, labels):
    return float(matthews_corrcoef(labels.astype(int), (logits >= 0).astype(int)))


def paired_boot_diff(la_logits, lb_logits, labels, metric, n_boot=10000, seed=42):
    """Paired bootstrap on matched (same val set) predictions.

    metric ∈ {'auroc','mcc'}. Returns (delta_mean, ci_lo, ci_hi, p_two_sided).
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        ya = labels[idx]
        # Skip degenerate resamples (only one class)
        if len(np.unique(ya)) < 2:
            deltas[b] = np.nan
            continue
        if metric == "auroc":
            ma = roc_auc_score(ya, la_logits[idx])
            mb = roc_auc_score(ya, lb_logits[idx])
        else:
            ma = matthews_corrcoef(ya, (la_logits[idx] >= 0).astype(int))
            mb = matthews_corrcoef(ya, (lb_logits[idx] >= 0).astype(int))
        deltas[b] = ma - mb
    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # 2-sided p ≈ 2 * min(P(Δ<0), P(Δ>0))
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(np.mean(deltas)), float(lo), float(hi), float(min(p, 1.0))


def collect_shards():
    rows = []
    for f in sorted(glob.glob(f"{SHARDS_DIR}/*.csv")):
        if os.path.getsize(f) == 0:
            continue
        try:
            df = pd.read_csv(f)
            rows.append(df)
        except Exception as e:
            print(f"  [skip {f}] {e}", file=sys.stderr)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def best_lr_rows(df):
    """Pick best LR per (task, ckpt, lora_config) by val AUROC."""
    return df.loc[df.groupby(["task", "ckpt", "lora_config"])["best_val_auroc"]
                    .idxmax()].reset_index(drop=True)


def add_mcc(df):
    """Compute MCC from saved per-sample preds."""
    mccs = []
    for _, r in df.iterrows():
        p = preds_path(r["task"], r["ckpt"], float(r["lr"]), r["lora_config"])
        if os.path.exists(p):
            d = np.load(p)
            mccs.append(round(mcc_from_logits(d["preds"], d["labels"]), 4))
        else:
            mccs.append(np.nan)
    df = df.copy()
    df["mcc"] = mccs
    return df


def compute_comparisons(best, cfg_filter):
    """Build pairwise rows with paired-bootstrap CIs."""
    pairs = [
        ("unlocked_ft",  "pretrained",   "unlocked_ft − pretrained (FT uplift)"),
        ("M_a300k",      "pretrained",   "M − pretrained (locked attacker boost)"),
        ("locked_no_ft", "pretrained",   "locked_no_ft − pretrained (clean lock test)"),
        ("M_a300k",      "unlocked_ft",  "M − unlocked_ft (lock marginal, CONFOUNDED)"),
    ]
    out_rows = []
    for task in sorted(best["task"].unique()):
        sub = best[(best.task == task) & (best.lora_config == cfg_filter)]
        ckpt_to_row = {r["ckpt"]: r for _, r in sub.iterrows()}
        for a, b, label in pairs:
            if a not in ckpt_to_row or b not in ckpt_to_row:
                continue
            ra, rb = ckpt_to_row[a], ckpt_to_row[b]
            pa = preds_path(task, a, float(ra["lr"]), cfg_filter)
            pb = preds_path(task, b, float(rb["lr"]), cfg_filter)
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            da, db = np.load(pa), np.load(pb)
            assert np.array_equal(da["labels"], db["labels"]), \
                f"label mismatch {a} vs {b} for {task}"
            labels = da["labels"]
            d_auroc, lo_a, hi_a, p_a = paired_boot_diff(
                da["preds"], db["preds"], labels, "auroc")
            d_mcc,   lo_m, hi_m, p_m = paired_boot_diff(
                da["preds"], db["preds"], labels, "mcc")
            out_rows.append(dict(
                task=task, lora_config=cfg_filter, comparison=label,
                ckpt_a=a, lr_a=ra["lr"], auroc_a=ra["best_val_auroc"], mcc_a=ra["mcc"],
                ckpt_b=b, lr_b=rb["lr"], auroc_b=rb["best_val_auroc"], mcc_b=rb["mcc"],
                d_auroc=round(d_auroc, 4),
                auroc_ci=f"[{lo_a:+.4f}, {hi_a:+.4f}]",
                p_auroc=round(p_a, 5),
                d_mcc=round(d_mcc, 4),
                mcc_ci=f"[{lo_m:+.4f}, {hi_m:+.4f}]",
                p_mcc=round(p_m, 5),
            ))
    return pd.DataFrame(out_rows)


def main():
    df = collect_shards()
    if df.empty:
        print("No shard CSVs found.", file=sys.stderr)
        return
    print(f"[collect] {len(df)} rows from {len(glob.glob(SHARDS_DIR + '/*.csv'))} shards")

    best = best_lr_rows(df)
    best = add_mcc(best)
    print(f"[best-LR] {len(best)} (task, ckpt, lora_config) combos")

    # Fair head-to-head: matched + force_locked, exclude SVD ckpts
    base_ckpts = ["pretrained", "locked_no_ft", "unlocked_ft", "M_a300k"]
    fair = best[best.ckpt.isin(base_ckpts) &
                best.lora_config.isin(["matched", "force_locked"])].copy()
    fair_out = f"{REPO}/results/hvue_lora_fair_headtohead.csv"
    fair.to_csv(fair_out, index=False)
    print(f"[write] {fair_out}  ({len(fair)} rows)")

    # SVD: matched config, svd ckpts only
    svd = best[best.ckpt.str.startswith("svd_")].copy()
    svd_out = f"{REPO}/results/hvue_lora_svd.csv"
    svd.to_csv(svd_out, index=False)
    print(f"[write] {svd_out}  ({len(svd)} rows)")

    # Pairwise comparisons for fair (matched)
    cmp_matched = compute_comparisons(best, "matched")
    cmp_force   = compute_comparisons(best, "force_locked")
    cmp = pd.concat([cmp_matched, cmp_force], ignore_index=True)
    cmp_out = f"{REPO}/results/hvue_lora_fair_pairwise.csv"
    cmp.to_csv(cmp_out, index=False)
    print(f"[write] {cmp_out}  ({len(cmp)} rows)")

    # Print readable summary
    print("\n=== Fair head-to-head (best-LR per ckpt × config × task) ===")
    cols = ["task", "ckpt", "lora_config", "lr",
            "best_val_auroc", "mcc", "n_lora_layers", "n_lora_specdef"]
    cols = [c for c in cols if c in fair.columns]
    print(fair[cols].sort_values(["task", "lora_config", "ckpt"]).to_string(index=False))

    print("\n=== Pairwise comparisons (paired bootstrap, 10k resamples) ===")
    print(cmp[["task", "lora_config", "comparison",
               "d_auroc", "auroc_ci", "p_auroc",
               "d_mcc",   "mcc_ci",   "p_mcc"]].to_string(index=False))

    print("\n=== SVD-chain (matched config) ===")
    if not svd.empty:
        scols = [c for c in cols if c in svd.columns]
        print(svd[scols].sort_values(["task", "ckpt"]).to_string(index=False))


if __name__ == "__main__":
    main()
