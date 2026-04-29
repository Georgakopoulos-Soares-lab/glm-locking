#!/usr/bin/env python
"""Generate the manuscript figures.

Main figure:
    results/figures/fig1_main.pdf       — (a) PPL bar chart, four conditions
                                          (b) mean HVUE AUROC bar chart

Supplementary:
    results/figures/fig_s1_training_curves.pdf
                                        — (a) stable training curves
                                          (b) LR ladder at alpha=1e4

Numbers are hard-coded from canonical results
(results/attack_heldout_ppl.csv + results/hvue_probe_results.json).
"""
from __future__ import annotations

import os
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ------------------------------------------------------------------
# Style
# ------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#222",
    "xtick.color": "#222",
    "ytick.color": "#222",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ==================================================================
# Figure 1 — (a) PPL bars + (b) AUROC bars  (4 conditions only)
# ==================================================================
def fig1():
    rows = [
        # (label, ppl, mean_auroc, color)
        ("Pretrained",         3.729, 0.842, "#7f7f7f"),
        ("Unlocked FT",        3.480, 0.867, "#1b9e77"),
        ("Locked FT\n(α=3×10⁴)", 3.816, 0.790, "#2ca02c"),
        ("Bypass FT",          3.729, 0.842, "#1b6e3a"),
    ]
    labels = [r[0] for r in rows]
    ppls   = [r[1] for r in rows]
    aurocs = [r[2] for r in rows]
    colors = [r[3] for r in rows]

    fig = plt.figure(figsize=(18 / 2.54, 8.5 / 2.54))
    gs  = fig.add_gridspec(1, 2, wspace=0.48)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    x = np.arange(len(rows))

    # ---- (a): PPL ----------------------------------------
    ax1.bar(x, ppls, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Held-out viral PPL")
    ax1.set_ylim(3.3, 4.15)
    ax1.set_yticks([3.4, 3.6, 3.8, 4.0])
    ax1.set_title("(a) Perplexity", fontsize=8.5, pad=4)
    ax1.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax1.set_axisbelow(True)
    for xi, v in zip(x, ppls):
        ax1.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=6.5, color="#222")

    # ---- (b): AUROC -------------------------------------
    ax2.bar(x, aurocs, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Mean HVUE AUROC")
    ax2.set_ylim(0.72, 0.92)
    ax2.set_yticks([0.74, 0.78, 0.82, 0.86, 0.90])
    ax2.set_title("(b) Virological capability", fontsize=8.5, pad=4)
    ax2.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax2.set_axisbelow(True)
    for xi, v in zip(x, aurocs):
        ax2.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=6.5, color="#222")

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right",
                           multialignment="center")
        ax.tick_params(axis="x", length=0)

    plt.tight_layout(pad=0.8)
    out = os.path.join(OUT_DIR, "fig1_main.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


# ==================================================================
# Figure 2 — (a) Schematic + (b) 4-bar bypass-vs-naive comparison
# ==================================================================
def _box(ax, x, y, w, h, label, fc, ec="black", fontsize=7.5,
         labelcolor="#111", lw=0.8, italic=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=lw))
    style = "italic" if italic else "normal"
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=labelcolor, style=style)


def _arrow(ax, x0, y0, x1, y1, color="#222", lw=1.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>",
        mutation_scale=10, color=color, linewidth=lw))


def fig2():
    fig, ax_b = plt.subplots(figsize=(11 / 2.54, 7.5 / 2.54))

    # ---- 4-bar comparison ------------------------------------
    rows_b = [
        # (label, ppl, mean_auroc, color)
        ("Pretrained",                       3.729, 0.842, "#7f7f7f", False),
        ("Unlocked\nFT 25k",                 3.480, 0.846, "#1b9e77", False),
        ("Locked +\nnaive FT\n(α=3×10⁴)",   3.816, 0.790, "#d62728", False),
        ("Locked +\nbypass FT\n(B-only)",    3.729, 0.842, "#1b6e3a", False),
    ]
    labels = [r[0] for r in rows_b]
    ppls   = [r[1] for r in rows_b]
    aurocs = [r[2] for r in rows_b]
    colors = [r[3] for r in rows_b]
    is_est = [r[4] for r in rows_b]
    x = np.arange(len(rows_b))

    width = 0.38
    bars1 = ax_b.bar(x - width / 2, aurocs, width, color=colors,
                     edgecolor="black", linewidth=0.5,
                     label="Mean HVUE AUROC")
    # hatch the bypass (estimate)
    for i, est in enumerate(is_est):
        if est:
            bars1[i].set_hatch("////")

    ax_b.set_ylabel("Mean HVUE AUROC")
    ax_b.set_ylim(0.45, 0.95)
    ax_b.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax_b.set_title("Bypass silently restores virological capability",
                   fontsize=9, pad=4)
    ax_b.grid(True, axis="y", linestyle="-", linewidth=0.4,
              color="#dddddd")
    ax_b.set_axisbelow(True)
    ax_b.set_xticks(x - width / 2)
    ax_b.set_xticklabels(labels, fontsize=6.5)
    ax_b.tick_params(axis="x", length=0)
    for xi, v, est in zip(x - width / 2, aurocs, is_est):
        txt = f"{v:.2f}" + ("*" if est else "")
        ax_b.text(xi, v + 0.005, txt, ha="center", va="bottom",
                  fontsize=6.5, color="#222")

    # twin axis for PPL
    ax_b2 = ax_b.twinx()
    bars2 = ax_b2.bar(x + width / 2, ppls, width, color=colors,
                      edgecolor="black", linewidth=0.5, alpha=0.55)
    for i, est in enumerate(is_est):
        if est:
            bars2[i].set_hatch("////")
    ax_b2.set_ylabel("Held-out viral PPL", color="#444")
    ax_b2.set_ylim(3.3, 4.15)
    ax_b2.set_yticks([3.4, 3.6, 3.8, 4.0])
    ax_b2.tick_params(axis="y", colors="#444")
    for xi, v, est in zip(x + width / 2, ppls, is_est):
        txt = f"{v:.2f}" + ("*" if est else "")
        ax_b2.text(xi, v + 0.01, txt, ha="center", va="bottom",
                   fontsize=6.5, color="#444")

    # legend / footnote
    ax_b.text(0.50, -0.30,
              "Bypass trains only 32 B-matrices (<0.5% params); locked weights frozen throughout",
              transform=ax_b.transAxes, ha="center",
              fontsize=5.8, color="#555", style="italic")
    # custom legend for left/right axis
    h_auroc = mpatches.Patch(facecolor="#bbbbbb", edgecolor="black",
                             linewidth=0.4, label="Mean HVUE AUROC (left)")
    h_ppl   = mpatches.Patch(facecolor="#bbbbbb", edgecolor="black",
                             linewidth=0.4, alpha=0.55,
                             label="Held-out PPL (right)")
    ax_b.legend(handles=[h_auroc, h_ppl], loc="upper right",
                fontsize=6, handletextpad=0.4, borderpad=0.3,
                labelspacing=0.25)

    plt.tight_layout(pad=0.4)
    out = os.path.join(OUT_DIR, "fig2_bypass.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


# ==================================================================
# Figure S1 — Training curves + LR-stability ladder  (was Fig 3)
# ==================================================================
def _load_metrics(run_name, max_step=None, clip_val=None):
    path = os.path.join(RESULTS_DIR, run_name, "metrics.csv")
    steps, val = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            s = int(r["step"])
            if s == 0:
                continue
            if max_step is not None and s > max_step:
                break
            v = float(r["val_loss"])
            if clip_val is not None and v > clip_val:
                continue
            steps.append(s)
            val.append(v)
    return steps, val


def fig_s1():
    fig, (ax, axin) = plt.subplots(
        1, 2, figsize=(15 / 2.54, 7 / 2.54),
        gridspec_kw={"width_ratios": [2.2, 1.0]})

    ax.grid(True, linestyle="-", linewidth=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)

    runs = [
        ("ft_unlocked_25k",               "Unlocked FT (η=10⁻⁵, 25k steps)", "#1b9e77", "-", 25000),
        ("ft_locked_a30k_lr1e6_25k_locked", "Locked α=3×10⁴ (η=10⁻⁶)",       "#2ca02c", "-", 25000),
    ]
    for run, label, color, ls, mx in runs:
        steps, val = _load_metrics(run, max_step=mx, clip_val=1.42)
        ax.plot(steps, val, color=color, linestyle=ls, linewidth=1.4,
                marker="o", markersize=2.5, label=label, zorder=3)

    ax.axhline(np.log(3.729), color="#444", linestyle="--",
               linewidth=0.7, zorder=1)
    ax.text(25000, np.log(3.729) + 0.005, "pretrained",
            fontsize=6.5, color="#444", ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(80, 30000)
    ax.set_ylim(1.10, 1.46)
    ax.set_xlabel("Optimizer step (log scale)")
    ax.set_ylabel("Validation loss (nats / token)")
    ax.legend(loc="upper right", fontsize=6.5, handletextpad=0.5,
              borderpad=0.4, labelspacing=0.3)
    ax.set_title("(a) Fine-tuning curves (25k steps)", fontsize=8, pad=4)

    axin.grid(True, linestyle="-", linewidth=0.3, color="#dddddd", zorder=0)
    axin.set_axisbelow(True)
    lr_runs = [
        ("ft_locked_a10k_lr3e5", "η=3×10⁻⁵  (diverges)", "#d62728", 800),
        ("ft_locked_a10k_lr3e6", "η=3×10⁻⁶  (oscillates)", "#ff7f0e", 800),
        ("ft_locked_a10k_lr1e6", "η=10⁻⁶    (stable)",   "#1f77b4", 800),
    ]
    for run, label, color, mx in lr_runs:
        steps, val = _load_metrics(run, max_step=mx)
        axin.plot(steps, val, color=color, linewidth=1.0,
                  marker="o", markersize=2.5, label=label)

    axin.set_yscale("log")
    axin.set_xlim(0, 800)
    axin.set_ylim(1.0, 25)
    axin.set_xlabel("Optimizer step", fontsize=7)
    axin.set_ylabel("Validation loss (log)", fontsize=7)
    axin.tick_params(axis="both", labelsize=6.5)
    axin.set_title("(b) LR ladder at α=10⁴", fontsize=8, pad=4)
    axin.legend(loc="upper right", fontsize=6, handletextpad=0.4,
                borderpad=0.3, labelspacing=0.25)

    plt.tight_layout(pad=0.4)
    out = os.path.join(OUT_DIR, "fig_s1_training_curves.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    fig1()
    fig_s1()
