"""
Compute AUROC + MCC table with 95% bootstrap CIs and pairwise significance tests
for the HVUE LoRA fine-tuning results.

Best-LR selection: max(best_val_auroc) per (ckpt, task) from hvue_lora_all.csv.
Bootstrap: 10 000 resamples (stratified by label), paired across models.
Significance: two-sided p-value = 2 × min(Pr[A−B <= 0], Pr[A−B >= 0]) under bootstrap.
MCC threshold: Youden's J on each bootstrap resample (honest evaluation).
Output: markdown table + CSV to results/hvue_lora_metrics_table.{md,csv}
"""

import os, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve

PRED_DIR  = "results/hvue_lora_preds"
CSV_PATH  = "results/hvue_lora_all.csv"
N_BOOT    = 10_000
RNG_SEED  = 42
OUT_MD    = "results/hvue_lora_metrics_table.md"
OUT_CSV   = "results/hvue_lora_metrics_table.csv"

TASKS = ["Host_Tropism", "Pathogenecity", "Transmissibility"]
CKPTS = ["pretrained", "locked_no_ft", "unlocked_ft", "M_a300k"]

CKPT_LABEL = {
    "pretrained":   "Pretrained",
    "locked_no_ft": "Locked (no FT)",
    "unlocked_ft":  "Unlocked FT",
    "M_a300k":      "M_α300k (locked FT)",
}
TASK_LABEL = {
    "Host_Tropism":    "Host Tropism",
    "Pathogenecity":   "Pathogenicity",
    "Transmissibility":"Transmissibility",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def mcc_at_youden(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmax(tpr - fpr)
    preds = (scores >= thresholds[idx]).astype(int)
    return matthews_corrcoef(labels, preds)

def bootstrap_metrics(labels, scores, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(labels)
    aurocs, mccs = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        l_b, s_b = labels[idx], scores[idx]
        if len(np.unique(l_b)) < 2:          # degenerate resample — skip
            aurocs[i] = np.nan
            mccs[i]   = np.nan
            continue
        aurocs[i] = roc_auc_score(l_b, s_b)
        mccs[i]   = mcc_at_youden(l_b, s_b)
    return aurocs, mccs

def pval_two_sided(boot_a, boot_b):
    """Paired bootstrap p-value (two-sided)."""
    diff = boot_a - boot_b
    diff = diff[~np.isnan(diff)]
    obs  = np.nanmean(diff)                  # observed mean difference
    p1   = np.mean(diff <= 0)
    p2   = np.mean(diff >= 0)
    return 2 * min(p1, p2)

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ── load best-LR prediction file per (ckpt, task) ────────────────────────────

df_csv = pd.read_csv(CSV_PATH)

def best_lr(task, ckpt):
    rows = df_csv[(df_csv.ckpt == ckpt) & (df_csv.task == task)]
    if rows.empty:
        return None
    best_row = rows.loc[rows.best_val_auroc.idxmax()]
    lr_str = f"{best_row.lr:.0e}".replace("e-0", "e-")
    return lr_str

def load_preds(task, ckpt):
    lr = best_lr(task, ckpt)
    if lr is None:
        return None, None
    fname = f"{task}_{ckpt}_{lr}.npz"
    path  = os.path.join(PRED_DIR, fname)
    if not os.path.exists(path):
        # try alternate zero-padding
        for f in os.listdir(PRED_DIR):
            if f.startswith(f"{task}_{ckpt}_") and f.endswith(".npz") and "__" not in f:
                path = os.path.join(PRED_DIR, f)
                break
        else:
            return None, None
    d = np.load(path)
    return d["preds"], d["labels"].astype(int)

# ── main computation ──────────────────────────────────────────────────────────

print("Computing metrics + bootstrap CIs …")
results = {}   # (task, ckpt) -> {auroc, mcc, auroc_ci, mcc_ci, boot_auroc, boot_mcc}

for task in TASKS:
    for ckpt in CKPTS:
        scores, labels = load_preds(task, ckpt)
        if scores is None:
            print(f"  MISSING: {task} × {ckpt}")
            continue
        auroc = roc_auc_score(labels, scores)
        mcc   = mcc_at_youden(labels, scores)
        b_auroc, b_mcc = bootstrap_metrics(labels, scores)
        results[(task, ckpt)] = dict(
            auroc=auroc, mcc=mcc,
            auroc_ci=(np.nanpercentile(b_auroc,2.5), np.nanpercentile(b_auroc,97.5)),
            mcc_ci  =(np.nanpercentile(b_mcc,  2.5), np.nanpercentile(b_mcc,  97.5)),
            boot_auroc=b_auroc, boot_mcc=b_mcc,
        )
        print(f"  {task:20s} {ckpt:20s}  AUROC={auroc:.4f}  MCC={mcc:.4f}")

# ── k-mer baseline ────────────────────────────────────────────────────────────
KMER = {"Host_Tropism": 0.903, "Pathogenecity": 0.846, "Transmissibility": 0.914}

# ── pairwise significance ─────────────────────────────────────────────────────
# Reference: pretrained; also M_a300k vs unlocked_ft
COMPARISONS = [
    ("unlocked_ft",  "pretrained",  "FT effect"),
    ("locked_no_ft", "pretrained",  "Lock-only effect"),
    ("M_a300k",      "pretrained",  "Locked-FT effect"),
    ("M_a300k",      "unlocked_ft", "Lock defense cost"),
]

sig_table = {}   # (task, ckpt_a, ckpt_b) -> {p_auroc, p_mcc}
for (ckpt_a, ckpt_b, label) in COMPARISONS:
    for task in TASKS:
        if (task, ckpt_a) not in results or (task, ckpt_b) not in results:
            continue
        p_auroc = pval_two_sided(results[(task, ckpt_a)]["boot_auroc"],
                                  results[(task, ckpt_b)]["boot_auroc"])
        p_mcc   = pval_two_sided(results[(task, ckpt_a)]["boot_mcc"],
                                  results[(task, ckpt_b)]["boot_mcc"])
        sig_table[(task, ckpt_a, ckpt_b)] = dict(p_auroc=p_auroc, p_mcc=p_mcc)

# ── build markdown ────────────────────────────────────────────────────────────

lines = []
lines.append("# HVUE LoRA Fine-Tuning: AUROC & MCC Table")
lines.append("")
lines.append("Predictions from best-LR run per model × task (full LoRA config, n=2000 val samples).")
lines.append("95% CI via stratified bootstrap (10 000 resamples). MCC at Youden-J threshold.")
lines.append("Significance: paired bootstrap two-sided p-value. \\*p<0.05, \\*\\*p<0.01, \\*\\*\\*p<0.001, ns.")
lines.append("")

# ── Table 1: AUROC ────────────────────────────────────────────────────────────
lines.append("## Table 1 — AUROC (best-LR per model)")
lines.append("")
header = "| Task | k-mer | " + " | ".join(CKPT_LABEL[c] for c in CKPTS) + " |"
sep    = "|:--|:--:|" + ":--:|" * len(CKPTS)
lines.append(header)
lines.append(sep)

for task in TASKS:
    km  = f"{KMER[task]:.3f}"
    cells = []
    vals  = []
    for ckpt in CKPTS:
        if (task, ckpt) not in results:
            cells.append("—")
            vals.append(-1)
        else:
            cells.append(results[(task, ckpt)]["auroc"])
            vals.append(results[(task, ckpt)]["auroc"])
    best_val = max(vals)
    row_cells = []
    for ckpt, v in zip(CKPTS, vals):
        if v < 0:
            row_cells.append("—")
            continue
        r = results[(task, ckpt)]
        ci_lo, ci_hi = r["auroc_ci"]
        cell = f"{v:.4f} [{ci_lo:.4f}–{ci_hi:.4f}]"
        if abs(v - best_val) < 1e-8:
            cell = f"**{cell}**"
        row_cells.append(cell)
    lines.append(f"| {TASK_LABEL[task]} | {km} | " + " | ".join(row_cells) + " |")

lines.append("")

# ── Table 2: MCC ──────────────────────────────────────────────────────────────
lines.append("## Table 2 — MCC (Youden-J threshold, best-LR per model)")
lines.append("")
lines.append(header)
lines.append(sep)

for task in TASKS:
    km  = "—"
    cells = []
    vals  = []
    for ckpt in CKPTS:
        if (task, ckpt) not in results:
            cells.append("—")
            vals.append(-99)
        else:
            cells.append(results[(task, ckpt)]["mcc"])
            vals.append(results[(task, ckpt)]["mcc"])
    best_val = max(vals)
    row_cells = []
    for ckpt, v in zip(CKPTS, vals):
        if v < -90:
            row_cells.append("—")
            continue
        r = results[(task, ckpt)]
        ci_lo, ci_hi = r["mcc_ci"]
        cell = f"{v:.4f} [{ci_lo:.4f}–{ci_hi:.4f}]"
        if abs(v - best_val) < 1e-8:
            cell = f"**{cell}**"
        row_cells.append(cell)
    lines.append(f"| {TASK_LABEL[task]} | — | " + " | ".join(row_cells) + " |")

lines.append("")

# ── Table 3: Pairwise significance ────────────────────────────────────────────
lines.append("## Table 3 — Pairwise Statistical Significance")
lines.append("")
lines.append("Paired bootstrap test (10 000 resamples, two-sided) on the same n=2000 validation set.")
lines.append("")
sig_header = "| Task | Comparison | ΔAUROC | p (AUROC) | sig | ΔMCC | p (MCC) | sig |"
sig_sep    = "|:--|:--|--:|--:|:--:|--:|--:|:--:|"
lines.append(sig_header)
lines.append(sig_sep)

for (ckpt_a, ckpt_b, label) in COMPARISONS:
    for task in TASKS:
        if (task, ckpt_a, ckpt_b) not in sig_table:
            continue
        r_a = results.get((task, ckpt_a))
        r_b = results.get((task, ckpt_b))
        if r_a is None or r_b is None:
            continue
        d_auroc = r_a["auroc"] - r_b["auroc"]
        d_mcc   = r_a["mcc"]   - r_b["mcc"]
        sig     = sig_table[(task, ckpt_a, ckpt_b)]
        p_au    = sig["p_auroc"]
        p_mc    = sig["p_mcc"]
        cmp_str = f"{CKPT_LABEL[ckpt_a]} vs {CKPT_LABEL[ckpt_b]}"
        line = (f"| {TASK_LABEL[task]} | {cmp_str} | "
                f"{d_auroc:+.4f} | {p_au:.4f} | {sig_stars(p_au)} | "
                f"{d_mcc:+.4f} | {p_mc:.4f} | {sig_stars(p_mc)} |")
        lines.append(line)

lines.append("")
lines.append(f"*Generated {pd.Timestamp.now().strftime('%Y-%m-%d')} — "
             f"bootstrap n_resamples={N_BOOT}, seed={RNG_SEED}*")

md_text = "\n".join(lines)

with open(OUT_MD, "w") as fh:
    fh.write(md_text)
print(f"\nMarkdown written to {OUT_MD}")

# ── CSV export ────────────────────────────────────────────────────────────────
rows_csv = []
for task in TASKS:
    for ckpt in CKPTS:
        if (task, ckpt) not in results:
            continue
        r = results[(task, ckpt)]
        row = dict(task=task, ckpt=ckpt,
                   auroc=r["auroc"],
                   auroc_ci_lo=r["auroc_ci"][0], auroc_ci_hi=r["auroc_ci"][1],
                   mcc=r["mcc"],
                   mcc_ci_lo=r["mcc_ci"][0],   mcc_ci_hi=r["mcc_ci"][1])
        # attach p-values vs pretrained and vs unlocked_ft
        for (ckpt_a, ckpt_b, _) in COMPARISONS:
            if ckpt == ckpt_a:
                k = (task, ckpt_a, ckpt_b)
                if k in sig_table:
                    row[f"p_auroc_vs_{ckpt_b}"] = sig_table[k]["p_auroc"]
                    row[f"p_mcc_vs_{ckpt_b}"]   = sig_table[k]["p_mcc"]
        rows_csv.append(row)

pd.DataFrame(rows_csv).to_csv(OUT_CSV, index=False)
print(f"CSV written to {OUT_CSV}")
print("\nDone.")
