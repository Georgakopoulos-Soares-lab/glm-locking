"""
Plot script: recomputes bootstrap (parallel) and draws AUROC/MCC figure
with 95% bootstrap CIs and paired significance brackets.
"""
import numpy as np, os, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve
from joblib import Parallel, delayed

OUT_PDF  = "results/hvue_lora_metrics_figure.pdf"
OUT_PNG  = "results/hvue_lora_metrics_figure.png"

TASKS  = ["Host_Tropism", "Pathogenecity", "Transmissibility"]
CKPTS  = ["pretrained", "locked_no_ft", "unlocked_ft", "M_a300k"]
TASK_LABEL = {"Host_Tropism":"Host Tropism","Pathogenecity":"Pathogenicity","Transmissibility":"Transmissibility"}
CKPT_LABEL = {"pretrained":"Pretrained","locked_no_ft":"Locked\n(no FT)","unlocked_ft":"Unlocked FT","M_a300k":"M_α300k\n(locked FT)"}
COLORS = {"pretrained":"#4C72B0","locked_no_ft":"#55A868","unlocked_ft":"#C44E52","M_a300k":"#DD8452"}
KMER = {"Host_Tropism":0.903,"Pathogenecity":0.846,"Transmissibility":0.914}
N_BOOT, SEED = 10000, 42
PRED_DIR = "results/hvue_lora_preds"
CSV_PATH = "results/hvue_lora_all.csv"

df_csv = pd.read_csv(CSV_PATH)
def best_lr_str(task, ckpt):
    rows = df_csv[(df_csv.ckpt == ckpt) & (df_csv.task == task)]
    if rows.empty: return None
    lr = rows.loc[rows.best_val_auroc.idxmax(), "lr"]
    return f"{lr:.0e}".replace("e-0", "e-")

def load_preds(task, ckpt):
    lr = best_lr_str(task, ckpt)
    if lr is None: return None, None
    fname = f"{task}_{ckpt}_{lr}.npz"
    path = os.path.join(PRED_DIR, fname)
    if not os.path.exists(path):
        for f in os.listdir(PRED_DIR):
            if f.startswith(f"{task}_{ckpt}_") and f.endswith(".npz") and "__" not in f:
                path = os.path.join(PRED_DIR, f); break
        else: return None, None
    d = np.load(path)
    return d["preds"], d["labels"].astype(int)

def _one_boot(labels, scores, seed):
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx = rng.integers(0, n, size=n)
    lb, sc = labels[idx], scores[idx]
    if len(np.unique(lb)) < 2: return np.nan, np.nan
    au = roc_auc_score(lb, sc)
    fpr, tpr, th = roc_curve(lb, sc)
    preds = (sc >= th[np.argmax(tpr - fpr)]).astype(int)
    mc = matthews_corrcoef(lb, preds)
    return au, mc

print("Loading predictions …")
all_data = {}
for task in TASKS:
    for ckpt in CKPTS:
        sc, lb = load_preds(task, ckpt)
        all_data[(task, ckpt)] = (sc, lb) if sc is not None else None

rng = np.random.default_rng(SEED)
seeds = rng.integers(0, 2**31, size=(len(TASKS), len(CKPTS), N_BOOT))

print("Running 12 × 10k bootstrap (parallel) …")
results = {}
for ti, task in enumerate(TASKS):
    for cj, ckpt in enumerate(CKPTS):
        entry = all_data[(task, ckpt)]
        if entry is None: results[(task, ckpt)] = None; continue
        sc, lb = entry
        out = Parallel(n_jobs=-1, verbose=0)(delayed(_one_boot)(lb, sc, int(s)) for s in seeds[ti, cj])
        au_arr = np.array([o[0] for o in out])
        mc_arr = np.array([o[1] for o in out])
        au = roc_auc_score(lb, sc)
        fpr, tpr, th = roc_curve(lb, sc)
        mc = matthews_corrcoef(lb, (sc >= th[np.argmax(tpr - fpr)]).astype(int))
        results[(task, ckpt)] = dict(auroc=au, mcc=mc,
            au_lo=np.nanpercentile(au_arr,2.5), au_hi=np.nanpercentile(au_arr,97.5),
            mc_lo=np.nanpercentile(mc_arr,2.5), mc_hi=np.nanpercentile(mc_arr,97.5),
            b_au=au_arr, b_mc=mc_arr)

def pval(ba, bb):
    d = ba - bb; d = d[~np.isnan(d)]
    return 2 * min(np.mean(d <= 0), np.mean(d >= 0))
def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

print("Plotting …")
fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey="row")
fig.subplots_adjust(hspace=0.52, wspace=0.28, top=0.90, bottom=0.08)
x = np.arange(len(CKPTS))
bw = 0.55
SIG_PAIRS = [("unlocked_ft","pretrained"),("M_a300k","pretrained"),("M_a300k","unlocked_ft")]

for col, task in enumerate(TASKS):
    for row, metric in enumerate(["auroc","mcc"]):
        ax = axes[row, col]
        vals2, lo2, hi2, b_arrays = [], [], [], []
        for ckpt in CKPTS:
            r = results.get((task, ckpt))
            if r is None: vals2.append(0); lo2.append(0); hi2.append(0); b_arrays.append(None); continue
            if metric == "auroc":
                v,l,h = r["auroc"], r["auroc"]-r["au_lo"], r["au_hi"]-r["auroc"]
                ba = r["b_au"]
            else:
                v,l,h = r["mcc"], r["mcc"]-r["mc_lo"], r["mc_hi"]-r["mcc"]
                ba = r["b_mc"]
            vals2.append(v); lo2.append(l); hi2.append(h); b_arrays.append(ba)

        ax.bar(x, vals2, width=bw, color=[COLORS[c] for c in CKPTS],
               edgecolor="white", linewidth=0.6,
               yerr=[lo2, hi2], capsize=3,
               error_kw=dict(elinewidth=0.8, ecolor="#333333"))

        if metric == "auroc":
            ax.axhline(KMER[task], color="#777777", lw=1.1, ls="--")

        all_v = [v for v in vals2 if v > 0]
        y_min = max(0, min(all_v)-0.12) if all_v else 0
        y_max = max(all_v)+0.10 if all_v else 1
        ax.set_ylim(y_min, y_max)
        pad = 0.003 if metric=="auroc" else 0.015
        bracket_lvl = 0
        for (ckpt_a, ckpt_b) in SIG_PAIRS:
            ia, ib = CKPTS.index(ckpt_a), CKPTS.index(ckpt_b)
            if b_arrays[ia] is None or b_arrays[ib] is None: continue
            p = pval(b_arrays[ia], b_arrays[ib])
            st = sig_stars(p)
            vi, vj = vals2[ia]+hi2[ia], vals2[ib]+hi2[ib]
            bh = max(vi, vj)+pad+bracket_lvl*pad*4
            if bh < y_max-pad*0.5:
                ax.plot([ia,ia,ib,ib],[vi+pad*0.3,bh,bh,vj+pad*0.3],color="#333333",lw=0.9)
                ax.text((ia+ib)/2, bh+pad*0.8, st, ha="center", va="bottom",
                        fontsize=7.5, color="#111" if st!="ns" else "#888")
                bracket_lvl += 1

        ax.set_xticks(x)
        ax.set_xticklabels([CKPT_LABEL[c] for c in CKPTS], fontsize=8.2, rotation=15, ha="right")
        if col == 0: ax.set_ylabel("AUROC" if metric=="auroc" else "MCC", fontsize=10)
        if row == 0: ax.set_title(TASK_LABEL[task], fontsize=11, fontweight="bold", pad=6)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

patches = [mpatches.Patch(color=COLORS[c], label=CKPT_LABEL[c].replace("\n"," ")) for c in CKPTS]
kmer_ln = Line2D([0],[0], color="#777777", lw=1.1, ls="--", label="k-mer baseline")
fig.legend(handles=patches+[kmer_ln], loc="upper center", ncol=5, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, 0.97))
fig.suptitle("HVUE LoRA Fine-Tuning — AUROC & MCC with 95% Bootstrap CI",
             fontsize=12, fontweight="bold", y=1.002)
fig.savefig(OUT_PDF, bbox_inches="tight", dpi=150)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved:\n  {OUT_PDF}\n  {OUT_PNG}")
