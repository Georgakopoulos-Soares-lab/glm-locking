"""
Build all 4 figures for the revised manuscript.
Run: python scripts/build_paper_figures.py
Output: paper/fig1_lora_bars.png, fig2_main_bars.png, fig3_kablation.png, figS1_probing_vs_lora.png
"""
import numpy as np, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.special import expit
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve, auc

# ── Data loading ──────────────────────────────────────────────────────────────
pred_dir = "results/hvue_lora_preds"

def load_best(task, ckpt, seed=0, config='full'):
    """Load best-LR predictions for (task, ckpt, seed, config).
    config: 'full' or 'matched' or 'force_locked'.
    For full config, tries multiple LRs and picks the one with highest AUROC."""
    cfg_tag = '' if config == 'full' else f'__{config}'
    best_au = -1
    best_data = None
    for lr in ['1e-04','5e-05','1e-05']:
        tag = '' if seed == 0 else f'_s{seed}'
        fname = f"{task}_{ckpt}_{lr}{cfg_tag}{tag}.npz"
        path = os.path.join(pred_dir, fname)
        if os.path.exists(path):
            d = np.load(path)
            probs = expit(d["preds"]); labels = d["labels"].astype(int)
            fpr, tpr, _ = roc_curve(labels, probs)
            au = auc(fpr, tpr)
            if au > best_au:
                best_au = au
                best_data = (probs, labels)
    return best_data

def compute_metrics(probs, labels):
    fpr, tpr, th = roc_curve(labels, probs)
    au = auc(fpr, tpr)
    mc = matthews_corrcoef(labels, (probs >= th[np.argmax(tpr-fpr)]).astype(int))
    return au, mc

# ── Significance brackets (sourced from manuscript Table 1) ───────────────────
# Paired bootstrap (n=10000) vs. pretrained, as reported in paper/main.tex Table 1.
# Keyed by (task, metric) -> {bar_index: stars}. Bar indices follow ckpts_fig1:
#   1 = unlocked_ft, 4 = M_a300k (naive full FT, strong lock).
# '' or 'ns' = not significant. Keep in sync with Table 1 if the table is revised.
SIG_VS_PRETRAINED = {
    ('Host_Tropism',     'au'): {1: 'ns',  4: '***'},
    ('Host_Tropism',     'mc'): {1: 'ns',  4: '***'},
    ('Pathogenecity',    'au'): {1: '***', 4: '***'},
    ('Pathogenecity',    'mc'): {1: '***', 4: '***'},
    ('Transmissibility', 'au'): {1: 'ns',  4: 'ns'},
    ('Transmissibility', 'mc'): {1: '**',  4: 'ns'},
}

def sig_bracket(ax, x1, x2, y, h, text):
    """Draw a significance bracket from x1→x2 at height y with tick height h."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c='#333333', clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h, text, ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333333', clip_on=False)

# ── Data for each figure ──────────────────────────────────────────────────────
tasks = ['Host_Tropism','Pathogenecity','Transmissibility']
task_short = {'Host_Tropism':'HT','Pathogenecity':'Path','Transmissibility':'Trans'}
ckpts_fig1 = ['pretrained','unlocked_ft','svd_k2_a30k','svd_k3_a30k','M_a300k','ft_lora_a10k']
labels_fig1 = ['Pretrained','Unlocked\nFT','SVD\nk=2','SVD\nk=3','Naive\nattack M','LoRA-\nlocked']
colors_fig1 = ['#4C72B0','#C44E52','#937860','#8C6BA8','#DD8452','#55A868']

kmer_au = [0.903, 0.846, 0.914]
kmer_mc = [0.662, 0.549, 0.733]

# =============================================================================
# FIGURE 1: LoRA-FT AUROC + MCC bars (full config, up to 4 seeds, 6 checkpoints)
# =============================================================================
print("Building fig1_lora_bars...")

# Collect data per (task, ckpt) across seeds — FULL config
fig1_data = {}  # (task, ckpt) -> {'au': [s0,s1,s2], 'mc': [s0,s1,s2]}
for task in tasks:
    for ckpt in ckpts_fig1:
        aus, mcs = [], []
        seeds_to_try = [0, 1, 2, 42]  # prefer 0,1,2; fall back to 42
        for seed in seeds_to_try:
            data = load_best(task, ckpt, seed, 'full')
            if data is not None:
                au, mc = compute_metrics(*data)
                aus.append(au); mcs.append(mc)
        fig1_data[(task, ckpt)] = {'au': aus, 'mc': mcs}

fig1, axes = plt.subplots(2, 3, figsize=(15, 8))
fig1.subplots_adjust(hspace=0.55, wspace=0.30, top=0.94)

for col, task in enumerate(tasks):
    for row, (metric, ylabel, kmer_vals) in enumerate([('au','AUROC', kmer_au), ('mc','MCC', kmer_mc)]):
        ax = axes[row, col]
        x = np.arange(len(ckpts_fig1))
        means, errs_lo, errs_hi = [], [], []
        for ckpt in ckpts_fig1:
            vals = fig1_data[(task, ckpt)][metric]
            m = np.mean(vals)
            means.append(m)
            if len(vals) > 1:
                errs_lo.append(m - min(vals))
                errs_hi.append(max(vals) - m)
            else:
                errs_lo.append(0); errs_hi.append(0)
        
        ax.bar(x, means, width=0.55, color=colors_fig1, edgecolor='white', linewidth=0.5,
               yerr=[errs_lo, errs_hi], capsize=4, 
               error_kw=dict(elinewidth=0.8, ecolor='#333333'))
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels_fig1, fontsize=6.8, rotation=0, ha='center')
        if col == 0: ax.set_ylabel(ylabel, fontsize=11)
        if row == 0: ax.set_title(task.replace('_',' '), fontsize=11, fontweight='bold', 
                                   loc='center', pad=6)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        
        # Set y-lim with padding
        all_v = [v for v in means if v > 0.2]
        ymin = max(0, min(all_v) - 0.12)
        data_max = max(all_v)
        span = max(data_max - ymin, 1e-6)
        tops = [means[i] + errs_hi[i] for i in range(len(ckpts_fig1))]
        ymax = data_max + 0.08

        # Significance brackets vs pretrained (idx 0): unlocked (idx 1), naive M (idx 4)
        # Stars taken from manuscript Table 1 (see SIG_VS_PRETRAINED above).
        sig = SIG_VS_PRETRAINED.get((task, metric), {})
        for (i, j), lvl in [((0, 1), 0.05), ((0, 4), 0.17)]:
            star = sig.get(j)
            if not star:
                continue
            y = max(max(tops[min(i, j):max(i, j) + 1]), data_max) + lvl * span
            sig_bracket(ax, i, j, y, 0.02 * span, star)
            ymax = max(ymax, y + 0.11 * span)
        ax.set_ylim(ymin, ymax)

# (Legend removed — checkpoints identified in LaTeX caption)
fig1.savefig('paper/fig1_lora_bars.png', bbox_inches='tight', dpi=150)
fig1.savefig('paper/fig1_lora_bars.pdf', bbox_inches='tight', dpi=150)
plt.close(fig1)
print("  → paper/fig1_lora_bars.png/pdf")

# =============================================================================
# FIGURE 2: PPL bars
# =============================================================================
print("Building fig2_main_bars...")

ppl_data = [
    ('Pretrained', 3.729, '#4C72B0'),
    ('Unlocked\nFT target', 3.487, '#C44E52'),
    ('M\n(α=3×10⁵)', 3.810, '#DD8452'),
    ('SVD k=2', 3.745, '#937860'),
    ('SVD k=3', 3.705, '#8C6BA8'),
    ('SVD k=5', 3.760, '#CCCCCC'),
    ('LoRA-locked\n(α=10⁴)', 3.729, '#55A868'),
]

fig2, ax = plt.subplots(figsize=(10, 5))
names = [d[0] for d in ppl_data]
vals = [d[1] for d in ppl_data]
cols = [d[2] for d in ppl_data]
x = np.arange(len(ppl_data))

ax.bar(x, vals, width=0.6, color=cols, edgecolor='white', linewidth=0.5)
ax.axhline(3.729, color='#4C72B0', lw=1.0, ls='--', alpha=0.6, label='Pretrained (3.73)')
ax.axhline(3.487, color='#C44E52', lw=1.0, ls='--', alpha=0.6, label='Unlocked FT target (3.49)')

# Value labels on bars
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel('Held-out viral PPL', fontsize=12)
ax.set_ylim(2.9, 4.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)
ax.legend(fontsize=9, frameon=False)

# fig2.suptitle removed — caption in LaTeX
fig2.tight_layout()
fig2.savefig('paper/fig2_main_bars.png', bbox_inches='tight', dpi=150)
fig2.savefig('paper/fig2_main_bars.pdf', bbox_inches='tight', dpi=150)
plt.close(fig2)
print("  → paper/fig2_main_bars.png/pdf")

# =============================================================================
# FIGURE 3: k-ablation — Pathogenicity AUROC (a) and PPL (b) for k=2,3,5
# =============================================================================
print("Building fig3_kablation...")

# Pathogenicity AUROC for k=2,3,5 — try full first, fall back to matched
k_aurocs = {}
for ckpt, k_label in [('svd_k2_a30k','2'), ('svd_k3_a30k','3'), ('svd_k5_a30k','5')]:
    data = load_best('Pathogenecity', ckpt, 0, 'full')
    if data is None:
        data = load_best('Pathogenecity', ckpt, 0, 'matched')  # k=5 only has matched
    if data is not None:
        au, mc = compute_metrics(*data)
        k_aurocs[k_label] = au

# PPL values
k_ppls = {'2': 3.745, '3': 3.705, '5': 3.760}
pretrained_ppl = 3.729

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
ks = ['2','3','5']
x3 = np.arange(3)
cols3 = ['#937860','#8C6BA8','#CCCCCC']

# (a) PPL — LEFT
ppl_vals = [k_ppls[k] for k in ks]
ax1.bar(x3, ppl_vals, width=0.5, color=cols3, edgecolor='white', linewidth=0.5)
ax1.axhline(pretrained_ppl, color='#4C72B0', lw=1.0, ls='--', label='Pretrained')
ax1.set_xticks(x3); ax1.set_xticklabels([f'k={k}' for k in ks])
ax1.set_ylabel('Held-out viral PPL', fontsize=11)
ax1.set_title('(a) Perplexity', fontsize=11, fontweight='bold', loc='center')
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
for i, v in enumerate(ppl_vals):
    ax1.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
ax1.set_ylim(min(ppl_vals)-0.02, max(ppl_vals)+0.05)
ax1.legend(fontsize=8, frameon=False)

# (b) AUROC — RIGHT
au_vals = [k_aurocs.get(k, 0) for k in ks]
ax2.bar(x3, au_vals, width=0.5, color=cols3, edgecolor='white', linewidth=0.5)
ax2.axhline(kmer_au[1], color='#777777', lw=1.0, ls='--', label='$k$-mer baseline')
ax2.set_xticks(x3); ax2.set_xticklabels([f'k={k}' for k in ks])
ax2.set_ylabel('Pathogenicity AUROC', fontsize=11)
ax2.set_title('(b) Downstream Capability', fontsize=11, fontweight='bold', loc='center')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
for i, v in enumerate(au_vals):
    ax2.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')
ylim1 = max(0.45, min(au_vals)-0.08)
ax2.set_ylim(ylim1, max(au_vals)+0.03)

# fig3.suptitle removed — caption in LaTeX
fig3.tight_layout()
fig3.savefig('paper/fig3_kablation.png', bbox_inches='tight', dpi=150)
fig3.savefig('paper/fig3_kablation.pdf', bbox_inches='tight', dpi=150)
plt.close(fig3)
print("  → paper/fig3_kablation.png/pdf")

# =============================================================================
# FIGURE S1: Probing vs LoRA FT — pretrained, unlocked, M
# =============================================================================
print("Building figS1_probing_vs_lora...")

# Probing data from hvue_probe.csv
probing = {
    # From hvue_probe.csv — final checkpoints
    ('pretrained','Host_Tropism'):       (0.860, 0.572),
    ('pretrained','Pathogenecity'):      (0.815, 0.455),
    ('pretrained','Transmissibility'):   (0.852, 0.538),
    ('unlocked_ft','Host_Tropism'):      (0.866, 0.579),  # ft_unlocked_25k_v2_unlocked
    ('unlocked_ft','Pathogenecity'):     (0.807, 0.459),
    ('unlocked_ft','Transmissibility'):  (0.871, 0.593),
    ('M_a300k','Host_Tropism'):          (0.873, 0.581),  # ft_locked_a300k_lr1e5_25k_locked
    ('M_a300k','Pathogenecity'):         (0.831, 0.490),
    ('M_a300k','Transmissibility'):      (0.871, 0.589),
}

# LoRA FT full config seed 0
lora_ft = {}
for task in tasks:
    for ckpt in ['pretrained','unlocked_ft','M_a300k']:
        data = load_best(task, ckpt, 0, 'full')
        if data is not None:
            au, mc = compute_metrics(*data)
            lora_ft[(ckpt, task)] = (au, mc)

ckpts_s1 = ['pretrained','unlocked_ft','M_a300k']
labels_s1 = ['Pretrained','Unlocked FT','Naive FT\n(α=3×10⁵)']
colors_s1 = ['#4C72B0','#C44E52','#DD8452']

figS1, axes = plt.subplots(2, 3, figsize=(15, 8))
figS1.subplots_adjust(hspace=0.45, wspace=0.22, top=0.92)

for col, task in enumerate(tasks):
    for row, (metric, ylabel, kmer_vals) in enumerate([('au','AUROC', kmer_au), ('mc','MCC', kmer_mc)]):
        ax = axes[row, col]
        x = np.arange(len(ckpts_s1)) * 3  # spacing
        bw = 1.0
        
        for i, ckpt in enumerate(ckpts_s1):
            # Probing bar
            prob_val = probing.get((ckpt, task), (None,None))[0 if metric=='au' else 1]
            # LoRA FT bar
            lora_val = lora_ft.get((ckpt, task), (None,None))[0 if metric=='au' else 1]
            
            if prob_val is not None:
                ax.bar(x[i] - bw/2, prob_val, width=bw*0.9, color=colors_s1[i], alpha=0.35, 
                       edgecolor=colors_s1[i], linewidth=0.8, hatch='//', label='Probing' if i==0 and row==0 and col==0 else '')
            if lora_val is not None:
                ax.bar(x[i] + bw/2, lora_val, width=bw*0.9, color=colors_s1[i], alpha=1.0,
                       edgecolor='white', linewidth=0.5, label='LoRA FT' if i==0 and row==0 and col==0 else '')
        
        ax.axhline(kmer_vals[col], color='#777777', lw=1.2, ls='--',
                   label='$k$-mer baseline' if i==0 and row==0 and col==0 else '')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_s1, fontsize=7.5, rotation=0, ha='center')
        if col == 0: ax.set_ylabel(ylabel, fontsize=11)
        if row == 0: ax.set_title(task.replace('_',' '), fontsize=11, fontweight='bold', loc='center')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

# Legend
p1 = Patch(facecolor='#888888', alpha=0.35, hatch='//', edgecolor='#888888', label='Probing (frozen features)')
p2 = Patch(facecolor='#4C72B0', alpha=1.0, label='LoRA FT (full config.)')
kmer_ln = Line2D([0],[0], color='#777777', lw=1.1, ls='--', label='$k$-mer baseline')
figS1.legend(handles=[p1, p2, kmer_ln], loc='upper center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.98))
# figS1.suptitle removed — caption in LaTeX
figS1.savefig('paper/figS1_probing_vs_lora.png', bbox_inches='tight', dpi=150)
figS1.savefig('paper/figS1_probing_vs_lora.pdf', bbox_inches='tight', dpi=150)
plt.close(figS1)
print("  → paper/figS1_probing_vs_lora.png/pdf")

print("\nAll 4 figures built. Output in paper/")
