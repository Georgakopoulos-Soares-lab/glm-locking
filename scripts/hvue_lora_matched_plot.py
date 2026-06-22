"""
Matched-config figure: 7 checkpoints, AUROC/MCC with 95% bootstrap CI
and paired significance brackets.
"""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.special import expit
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve, auc
from joblib import Parallel, delayed

OUT_PDF  = "results/hvue_lora_matched_figure.pdf"
OUT_PNG  = "results/hvue_lora_matched_figure.png"

TASKS  = ["Host_Tropism", "Pathogenecity", "Transmissibility"]
CKPTS  = ["pretrained", "unlocked_ft", "locked_no_ft", "M_a300k",
          "svd_k3_a30k", "svd_k2_a30k", "svd_k5_a30k"]

TASK_LABEL = {"Host_Tropism":"Host Tropism",
              "Pathogenecity":"Pathogenicity",
              "Transmissibility":"Transmissibility"}

CKPT_LABEL = {"pretrained":"Pretrained",
              "unlocked_ft":"Unlocked\nFT",
              "locked_no_ft":"Locked\n(no FT)",
              "M_a300k":"M_α300k\n(locked FT)",
              "svd_k3_a30k":"SVD k=3\n(locked FT)",
              "svd_k2_a30k":"SVD k=2\n(locked FT)",
              "svd_k5_a30k":"SVD k=5\n(locked FT)"}

COLORS = {"pretrained":"#4C72B0",
          "unlocked_ft":"#C44E52",
          "locked_no_ft":"#55A868",
          "M_a300k":"#DD8452",
          "svd_k3_a30k":"#937860",
          "svd_k2_a30k":"#8C6BA8",
          "svd_k5_a30k":"#CCCCCC"}

KMER = {"Host_Tropism":0.903, "Pathogenecity":0.846, "Transmissibility":0.914}
N_BOOT, SEED = 10000, 42
PRED_DIR = "results/hvue_lora_preds"

# ------------------------------------------------------------
# Known best-LR files per (task, ckpt) – matched config only
# ------------------------------------------------------------
BEST_FILES = {
    ("Host_Tropism","pretrained"):    "Host_Tropism_pretrained_1e-04__matched.npz",
    ("Host_Tropism","unlocked_ft"):   "Host_Tropism_unlocked_ft_1e-04__matched.npz",
    ("Host_Tropism","locked_no_ft"):  "Host_Tropism_locked_no_ft_1e-04__matched.npz",
    ("Host_Tropism","M_a300k"):       "Host_Tropism_M_a300k_5e-05__matched.npz",
    ("Host_Tropism","svd_k3_a30k"):   "Host_Tropism_svd_k3_a30k_1e-04__matched.npz",
    ("Host_Tropism","svd_k2_a30k"):   "Host_Tropism_svd_k2_a30k_1e-04__matched.npz",
    ("Host_Tropism","svd_k5_a30k"):   "Host_Tropism_svd_k5_a30k_5e-05__matched.npz",
    ("Pathogenecity","pretrained"):   "Pathogenecity_pretrained_1e-04__matched.npz",
    ("Pathogenecity","unlocked_ft"):  "Pathogenecity_unlocked_ft_1e-04__matched.npz",
    ("Pathogenecity","locked_no_ft"): "Pathogenecity_locked_no_ft_1e-04__matched.npz",
    ("Pathogenecity","M_a300k"):      "Pathogenecity_M_a300k_1e-04__matched.npz",
    ("Pathogenecity","svd_k3_a30k"):  "Pathogenecity_svd_k3_a30k_1e-04__matched.npz",
    ("Pathogenecity","svd_k2_a30k"):  "Pathogenecity_svd_k2_a30k_5e-05__matched.npz",
    ("Pathogenecity","svd_k5_a30k"):  "Pathogenecity_svd_k5_a30k_1e-04__matched.npz",
    ("Transmissibility","pretrained"):   "Transmissibility_pretrained_5e-05__matched.npz",
    ("Transmissibility","unlocked_ft"):  "Transmissibility_unlocked_ft_1e-04__matched.npz",
    ("Transmissibility","locked_no_ft"): "Transmissibility_locked_no_ft_5e-05__matched.npz",
    ("Transmissibility","M_a300k"):      "Transmissibility_M_a300k_5e-05__matched.npz",
    ("Transmissibility","svd_k3_a30k"):  "Transmissibility_svd_k3_a30k_1e-04__matched.npz",
    ("Transmissibility","svd_k2_a30k"):  "Transmissibility_svd_k2_a30k_1e-04__matched.npz",
    ("Transmissibility","svd_k5_a30k"):  "Transmissibility_svd_k5_a30k_1e-04__matched.npz",
}

def load_preds(task, ckpt):
    fname = BEST_FILES.get((task, ckpt))
    if fname is None: return None, None
    path = os.path.join(PRED_DIR, fname)
    if not os.path.exists(path): return None, None
    d = np.load(path)
    logits = d["preds"]
    probs = expit(logits)
    return probs, d["labels"].astype(int)

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

print("Loading predictions (matched config) …")
all_data = {}
for task in TASKS:
    for ckpt in CKPTS:
        sc, lb = load_preds(task, ckpt)
        all_data[(task, ckpt)] = (sc, lb) if sc is not None else None

rng = np.random.default_rng(SEED)
n_pairs = len(TASKS) * len(CKPTS)
seeds = rng.integers(0, 2**31, size=(n_pairs, N_BOOT))

print(f"Running {n_pairs} × {N_BOOT} bootstrap (parallel) …")
results = {}
pair_idx = 0
for task in TASKS:
    for ckpt in CKPTS:
        entry = all_data[(task, ckpt)]
        if entry is None:
            results[(task, ckpt)] = None
            pair_idx += 1
            continue
        sc, lb = entry
        out = Parallel(n_jobs=-1, verbose=0)(
            delayed(_one_boot)(lb, sc, int(s)) for s in seeds[pair_idx])
        au_arr = np.array([o[0] for o in out])
        mc_arr = np.array([o[1] for o in out])
        au = roc_auc_score(lb, sc)
        fpr, tpr, th = roc_curve(lb, sc)
        mc = matthews_corrcoef(lb, (sc >= th[np.argmax(tpr - fpr)]).astype(int))
        results[(task, ckpt)] = dict(
            auroc=au, mcc=mc,
            au_lo=np.nanpercentile(au_arr, 2.5), au_hi=np.nanpercentile(au_arr, 97.5),
            mc_lo=np.nanpercentile(mc_arr, 2.5), mc_hi=np.nanpercentile(mc_arr, 97.5),
            b_au=au_arr, b_mc=mc_arr)
        pair_idx += 1

def pval(ba, bb):
    d = ba - bb
    d = d[~np.isnan(d)]
    if len(d) == 0: return 1.0
    return 2 * min(np.mean(d <= 0), np.mean(d >= 0))

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# Key significance pairs to show
SIG_PAIRS = [
    ("svd_k3_a30k", "pretrained"),
    ("svd_k3_a30k", "locked_no_ft"),
    ("svd_k3_a30k", "unlocked_ft"),
    ("svd_k2_a30k", "pretrained"),
    ("svd_k2_a30k", "locked_no_ft"),
    ("svd_k2_a30k", "unlocked_ft"),
    ("unlocked_ft", "pretrained"),
    ("M_a300k", "locked_no_ft"),
]

print("Plotting …")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.subplots_adjust(hspace=0.55, wspace=0.30, top=0.90, bottom=0.10)
x = np.arange(len(CKPTS))
bw = 0.65

# Only show svd_k5 if its value is > 0.5 (otherwise it squashes y-axis)
DROP_K5_TASKS = set()

for col, task in enumerate(TASKS):
    for row, metric in enumerate(["auroc", "mcc"]):
        ax = axes[row, col]
        vals2, lo2, hi2, b_arrays, ckpts_present = [], [], [], [], []
        
        for ckpt in CKPTS:
            # Skip svd_k5 for Host_Tropism & Pathogenecity if it squashes
            r = results.get((task, ckpt))
            if r is None:
                vals2.append(np.nan); lo2.append(np.nan); hi2.append(np.nan)
                b_arrays.append(None); ckpts_present.append(ckpt)
                continue
            if metric == "auroc":
                v, l, h = r["auroc"], r["auroc"]-r["au_lo"], r["au_hi"]-r["auroc"]
                ba = r["b_au"]
            else:
                v, l, h = r["mcc"], r["mcc"]-r["mc_lo"], r["mc_hi"]-r["mcc"]
                ba = r["b_mc"]
            vals2.append(v); lo2.append(l); hi2.append(h)
            b_arrays.append(ba); ckpts_present.append(ckpt)

        # Filter out NaN
        valid_idx = [i for i, v in enumerate(vals2) if not np.isnan(v)]
        x_valid = [x[i] for i in valid_idx]
        v_valid = [vals2[i] for i in valid_idx]
        l_valid = [lo2[i] for i in valid_idx]
        h_valid = [hi2[i] for i in valid_idx]
        c_valid = [COLORS[CKPTS[i]] for i in valid_idx]
        
        ax.bar(x_valid, v_valid, width=bw, color=c_valid,
               edgecolor="white", linewidth=0.6,
               yerr=[l_valid, h_valid], capsize=3,
               error_kw=dict(elinewidth=0.8, ecolor="#333333"))

        if metric == "auroc":
            ax.axhline(KMER[task], color="#777777", lw=1.1, ls="--")

        all_v = [v for v in v_valid if v > 0]
        y_min = max(0, min(all_v)-0.10) if all_v else 0
        y_max = max(all_v)+0.10 if all_v else 1
        ax.set_ylim(y_min, y_max)
        
        # Build lookup for bracket positions
        v_dict = {CKPTS[i]: vals2[i] for i in valid_idx}
        h_dict = {CKPTS[i]: hi2[i] for i in valid_idx}
        b_dict = {CKPTS[i]: b_arrays[i] for i in valid_idx}
        
        pad = 0.005 if metric == "auroc" else 0.020
        bracket_lvl = 0
        for (ckpt_a, ckpt_b) in SIG_PAIRS:
            if ckpt_a not in v_dict or ckpt_b not in v_dict: continue
            ba, bb = b_dict[ckpt_a], b_dict[ckpt_b]
            if ba is None or bb is None: continue
            p = pval(ba, bb)
            st = sig_stars(p)
            ia = [i for i, c in enumerate(CKPTS) if c == ckpt_a][0]
            ib = [i for i, c in enumerate(CKPTS) if c == ckpt_b][0]
            vi = v_dict[ckpt_a] + h_dict[ckpt_a]
            vj = v_dict[ckpt_b] + h_dict[ckpt_b]
            bh = max(vi, vj) + pad + bracket_lvl * pad * 5
            if bh < y_max - pad * 0.5:
                ax.plot([ia, ia, ib, ib],
                        [vi+pad*0.3, bh, bh, vj+pad*0.3],
                        color="#333333", lw=0.9)
                ax.text((ia+ib)/2, bh+pad*0.8, st, ha="center", va="bottom",
                        fontsize=7, color="#111" if st!="ns" else "#888")
                bracket_lvl += 1

        ax.set_xticks(x)
        ax.set_xticklabels([CKPT_LABEL[c] for c in CKPTS], fontsize=7.5, rotation=20, ha="right")
        if col == 0:
            ax.set_ylabel("AUROC" if metric == "auroc" else "MCC", fontsize=11)
        if row == 0:
            ax.set_title(TASK_LABEL[task], fontsize=12, fontweight="bold", pad=6)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

# Legend
patches = [mpatches.Patch(color=COLORS[c], label=CKPT_LABEL[c].replace("\n"," ")) for c in CKPTS]
kmer_ln = Line2D([0],[0], color="#777777", lw=1.1, ls="--", label="k-mer baseline")
fig.legend(handles=patches+[kmer_ln], loc="upper center", ncol=8, fontsize=7.5,
           frameon=False, bbox_to_anchor=(0.5, 0.97))
fig.suptitle("HVUE LoRA Fine-Tuning — Matched Config (96 layers)\nAUROC & MCC with 95% Bootstrap CI & Paired Significance",
             fontsize=13, fontweight="bold", y=1.005)
fig.savefig(OUT_PDF, bbox_inches="tight", dpi=150)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved:\n  {OUT_PDF}\n  {OUT_PNG}")
