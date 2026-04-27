#!/usr/bin/env python
"""Generate the manuscript figures.

Main figures (2):
    results/figures/fig1_main.pdf       — (a) PPL vs AUROC paired bars
                                          (b) dose-response across alpha
    results/figures/fig2_bypass.pdf     — (a) SpecDef + bypass schematic
                                          (b) 4-bar comparison

Supplementary:
    results/figures/fig_s1_training_curves.pdf
                                        — (a) stable training curves
                                          (b) LR ladder at alpha=1e4

Numbers are hard-coded from canonical results
(results/attack_heldout_ppl.csv + results/hvue_probe.csv).
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
# Figure 1 — (a) Paired bars + (b) Dose-response
# ==================================================================
def fig1():
    rows = [
        # (label, ppl, mean_auroc, color)
        ("Pretrained",         3.729, 0.842, "#7f7f7f"),
        ("Unlocked\nFT 25k",   3.480, 0.846, "#1b9e77"),
        ("α=10⁴",              3.996, 0.667, "#1f77b4"),
        ("α=3×10⁴",            4.012, 0.601, "#2ca02c"),
        ("α=10⁵",              3.991, 0.592, "#ff7f0e"),
        ("α=10⁶",              4.009, 0.520, "#d62728"),
    ]
    labels = [r[0] for r in rows]
    ppls   = [r[1] for r in rows]
    aurocs = [r[2] for r in rows]
    colors = [r[3] for r in rows]

    fig = plt.figure(figsize=(18 / 2.54, 7.2 / 2.54))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.1], wspace=0.45)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    x = np.arange(len(rows))

    # ---- (a)-left: PPL ----------------------------------------
    ax1.bar(x, ppls, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Held-out viral PPL")
    ax1.set_ylim(3.3, 4.15)
    ax1.set_yticks([3.4, 3.6, 3.8, 4.0])
    ax1.set_title("(a) Perplexity", fontsize=8.5, pad=4)
    ax1.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax1.set_axisbelow(True)
    for xi, v in zip(x, ppls):
        ax1.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=6.2, color="#222")

    # ---- (a)-right: AUROC -------------------------------------
    ax2.bar(x, aurocs, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Mean HVUE AUROC")
    ax2.set_ylim(0.45, 0.95)
    ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax2.set_title("(b) Virological capability", fontsize=8.5, pad=4)
    ax2.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax2.set_axisbelow(True)
    for xi, v in zip(x, aurocs):
        ax2.text(xi, v + 0.005, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=6.2, color="#222")

    short_x = ["Pretrained", "Unlocked\nFT 25k",
               "α=10⁴", "α=3×10⁴", "α=10⁵", "α=10⁶"]
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(short_x, fontsize=6.5, rotation=30, ha="right")
        ax.tick_params(axis="x", length=0)

    # ---- (c): dose-response per task --------------------------
    x_labels = ["Baseline", "10⁴", "3×10⁴", "10⁵", "10⁶"]
    xd = np.arange(len(x_labels))

    tasks = {
        "Host Tropism":      dict(y=[0.860, 0.625, 0.544, 0.536, 0.531],
                                  color="#1f77b4", marker="o"),
        "Pathogenicity":     dict(y=[0.815, 0.641, 0.588, 0.558, 0.505],
                                  color="#ff7f0e", marker="s"),
        "Transmissibility":  dict(y=[0.852, 0.734, 0.670, 0.682, 0.525],
                                  color="#2ca02c", marker="^"),
    }
    ax3.grid(True, axis="y", linestyle="-", linewidth=0.4,
             color="#dddddd", zorder=0)
    ax3.set_axisbelow(True)
    ax3.axhspan(0.50, 0.55, color="#cccccc", alpha=0.30, zorder=0)
    ax3.axhline(0.5, color="#d62728", linestyle="--", linewidth=0.7, zorder=1)
    ax3.text(xd[-1] + 0.05, 0.505, "Chance", ha="right", va="bottom",
             fontsize=6.5, color="#a33", style="italic")
    for name, spec in tasks.items():
        ax3.plot(xd, spec["y"], color=spec["color"], marker=spec["marker"],
                 markersize=4.5, linewidth=1.3, label=name,
                 markeredgecolor="black", markeredgewidth=0.4, zorder=3)
    ax3.set_xticks(xd)
    ax3.set_xticklabels(x_labels, fontsize=7)
    ax3.set_xlabel("Lock strength α", fontsize=8)
    ax3.set_ylabel("AUROC")
    ax3.set_ylim(0.45, 0.90)
    ax3.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax3.set_title("(c) Per-task dose-response", fontsize=8.5, pad=4)
    ax3.legend(loc="upper right", fontsize=6.5, handletextpad=0.4,
               borderpad=0.3, labelspacing=0.3)

    plt.tight_layout(pad=0.4)
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
        ("Pretrained",                 3.729, 0.842, "#7f7f7f", False),
        ("Unlocked\nFT 25k",           3.480, 0.846, "#1b9e77", False),
        ("Locked +\nnaive FT\n(α=10⁴)", 3.996, 0.667, "#d62728", False),
        ("Locked +\nbypass FT\n(est.)", 3.49,  0.84,  "#1b6e3a", True),
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
    ax_b.set_title("Bypass recovers what naive FT loses",
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
              "* bypass row estimated from Rosati et al. (2026), Theorem 8",
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
        ("ft_unlocked_25k",                "Unlocked FT (η=10⁻⁵, 25k st.)", "#1b9e77", "-", 25000),
        ("ft_locked_a10k_lr1e6_25k_locked",  "Locked α=10⁴  (η=10⁻⁶)",         "#1f77b4", "-", 25000),
        ("ft_locked_a30k_lr1e6_25k_locked",  "Locked α=3×10⁴ (η=10⁻⁶)",        "#2ca02c", "-", 25000),
        ("ft_locked_a100k_lr1e6_25k_locked", "Locked α=10⁵  (η=10⁻⁶)",         "#ff7f0e", "-", 25000),
        ("ft_locked_a1M_lr1e7_25k_locked",   "Locked α=10⁶  (η=10⁻⁷)",         "#d62728", "-", 25000),
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
    ax.legend(loc="upper right", fontsize=6.2, handletextpad=0.5,
              borderpad=0.4, labelspacing=0.3)
    ax.set_title("(a) Fine-tuning curves (25k steps each)", fontsize=8, pad=4)

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
    fig2()
    fig_s1()
