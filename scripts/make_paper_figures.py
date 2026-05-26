#!/usr/bin/env python
"""Generate all manuscript figures for the glm-locking paper.

Outputs (all written to paper/ directory):
    fig1_pareto.{pdf,png}         — PPL vs AUROC Pareto scatter (all conditions)
    fig2_main_bars.{pdf,png}      — (a) PPL bars  (b) AUROC bars (all complete conditions)
    fig3_kablation.{pdf,png}      — SVD-chain k and α ablation
    fig_s1_training_curves.{pdf,png} — Training dynamics + LR stability

All numbers sourced directly from
    results/attack_heldout_ppl.csv
    results/hvue_probe.csv
"""
from __future__ import annotations

import os
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
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

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "paper")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Canonical results (from results/attack_heldout_ppl.csv and
# results/hvue_probe.csv — verified 2025-05)
# fmt: (label, ppl, tropism, pathog, trans, color, marker, hatch)
# ------------------------------------------------------------------
PRETRAINED_PPL  = 3.7289
PRETRAINED_AUROC = 0.8419
UNLOCKED_PPL    = 3.4865
UNLOCKED_AUROC  = 0.867   # unlocked fine-tuning ceiling (Table 1)

ALL_CONDITIONS = [
    # (label_short, label_long, ppl, trop, path, trans, color, marker, hatch, group)
    ("Pretrained",
     "Pretrained\n(zero-shot)",
     3.7289, 0.8596, 0.8146, 0.8516,
     "#7f7f7f", "s", None, "baseline"),

    ("Unlocked FT",
     "Unlocked FT\n(25k steps)",
     3.4865, 0.876, 0.842, 0.871,
     "#1b9e77", "D", None, "baseline"),

    ("Naive FT\nα=10⁴",
     "Locked naive FT\n(α=10⁴)",
     3.7533, 0.8640, 0.8178, 0.8617,
     "#e6ab02", "^", None, "naive"),

    ("Naive FT\nα=3×10⁴",
     "Locked naive FT η=10⁻⁶\n(α=3×10⁴)",
     3.8157, 0.7972, 0.7883, 0.7851,
     "#d62728", "v", None, "naive"),

    ("Naive FT η=10⁻⁵\nα=3×10⁴",
     "Locked naive FT η=10⁻⁵\n(α=3×10⁴)",
     3.7756, 0.9045, 0.8471, 0.8939,
     "#b22222", "v", None, "naive"),

    ("LoRA\nα=10⁴",
     "LoRA r=16\n(α=10⁴)",
     3.7290, 0.8389, 0.7659, 0.7788,
     "#ff7f0e", "P", None, "lora"),

    ("B-bypass\nα=10⁴",
     "B-injection bypass\n(α=10⁴)",
     4.0913, 0.6883, 0.7353, 0.8453,
     "#9467bd", "X", "////", "bypass"),

    ("B-bypass\nα=3×10⁴",
     "B-injection bypass\n(α=3×10⁴)",
     4.1228, 0.7070, 0.7345, 0.8478,
     "#7b2d8b", "X", "////", "bypass"),

    ("SVD k=3\nα=10⁴",
     "SVD-chain k=3\n(α=10⁴)",
     3.6670, 0.8173, 0.7819, 0.8188,
     "#aec7e8", "o", None, "svd"),

    ("SVD k=2\nα=3×10⁴",
     "SVD-chain k=2\n(α=3×10⁴)",
     3.7450, 0.8454, 0.7862, 0.8404,
     "#4a90c2", "o", None, "svd"),

    ("SVD k=3\nα=3×10⁴",
     "SVD-chain k=3\n(α=3×10⁴)",
     3.7051, 0.8584, 0.8032, 0.8454,
     "#005f9e", "o", None, "svd"),

    ("SVD k=5\nα=3×10⁴",
     "SVD-chain k=5\n(α=3×10⁴)",
     3.7603, 0.8519, 0.8075, 0.8451,
     "#003a6b", "o", None, "svd"),
]

def _mean(trop, path, trans):
    return (trop + path + trans) / 3.0

def _save(name):
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"{name}.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close()


# ==================================================================
# Figure 1 — Parallel-coordinate plot: PPL (left axis) vs mean AUROC (right axis)
# ==================================================================
def fig1_pareto():
    """Parallel-coordinate comparison of all conditions.

    Left axis:  Held-out viral PPL  (inverted: lower PPL = higher y = better)
    Right axis: Mean HVUE AUROC     (higher AUROC = higher y = better)
    Each condition is a line from its PPL position to its AUROC position.
    Lines sloping upward left-to-right trade generative quality for AUROC.
    """
    PPL_BOT, PPL_TOP = 4.30, 3.28   # y=0 worst PPL, y=1 best PPL
    AUC_BOT, AUC_TOP = 0.71, 0.92

    def ppl_norm(v):  return (PPL_BOT - v) / (PPL_BOT - PPL_TOP)
    def auc_norm(v):  return (v - AUC_BOT) / (AUC_TOP - AUC_BOT)

    # (name, ppl, mean_auroc, color, lw, group)
    conds = [
        ("Pretrained",            PRETRAINED_PPL,  PRETRAINED_AUROC,            "#7f7f7f", 1.4, "baseline"),
        ("Unlocked FT",           UNLOCKED_PPL,    UNLOCKED_AUROC,              "#1b9e77", 2.2, "baseline"),
        ("A: Naive (α=10⁴)",      3.7533, _mean(0.8640, 0.8178, 0.8617),        "#e6ab02", 1.0, "naive"),
        ("B: Naive η=10⁻⁶ (α=3×10⁴)",   3.8157, _mean(0.7972, 0.7883, 0.7851),        "#d62728", 1.4, "naive"),
        ("C: Naive η=10⁻⁵ (α=3×10⁴)",   3.7756, _mean(0.9045, 0.8471, 0.8939),        "#b22222", 1.8, "naive"),
        ("D: LoRA r=16",          3.7290, _mean(0.8389, 0.7659, 0.7788),        "#ff7f0e", 1.0, "lora"),
        ("E: Bypass (α=10⁴)",     4.0913, _mean(0.6883, 0.7353, 0.8453),        "#9467bd", 1.0, "bypass"),
        ("F: Bypass (α=3×10⁴)",  4.1228, _mean(0.7070, 0.7345, 0.8478),        "#7b2d8b", 1.0, "bypass"),
        ("G: SVD k=3 (α=10⁴)",   3.6670, _mean(0.8173, 0.7819, 0.8188),        "#aec7e8", 1.2, "svd"),
        ("H: SVD k=2 (α=3×10⁴)", 3.7450, _mean(0.8454, 0.7862, 0.8404),        "#4a90c2", 1.4, "svd"),
        ("I: SVD k=3 (α=3×10⁴)", 3.7051, _mean(0.8584, 0.8032, 0.8454),        "#005f9e", 1.8, "svd"),
        ("J: SVD k=5 (α=3×10⁴)", 3.7603, _mean(0.8519, 0.8075, 0.8451),        "#003a6b", 1.4, "svd"),
    ]

    fig, ax = plt.subplots(figsize=(15 / 2.54, 13 / 2.54))
    ax.set_xlim(-0.30, 1.65)
    ax.set_ylim(-0.10, 1.24)
    ax.axis("off")

    # Axis bars
    ax.plot([0, 0], [0, 1], color="#333", lw=1.5, zorder=6)
    ax.plot([1, 1], [0, 1], color="#333", lw=1.5, zorder=6)

    # PPL ticks (left side; lower PPL → higher y position)
    for v in [3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2]:
        y = ppl_norm(v)
        if -0.03 <= y <= 1.05:
            ax.plot([-0.018, 0], [y, y], color="#444", lw=0.7)
            ax.text(-0.025, y, f"{v:.1f}", ha="right", va="center", fontsize=7.0)

    # AUROC ticks (right side)
    for v in [0.72, 0.76, 0.80, 0.84, 0.88]:
        y = auc_norm(v)
        ax.plot([1, 1.018], [y, y], color="#444", lw=0.7)
        ax.text(1.025, y, f"{v:.2f}", ha="left", va="center", fontsize=7.0)

    # Axis labels
    ax.text(0, 1.14, "Viral PPL", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold")
    ax.text(0, 1.08, "(lower is better ↓)", ha="center", va="bottom",
            fontsize=6.2, color="#555")
    ax.text(1, 1.14, "Mean HVUE AUROC", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold")
    ax.text(1, 1.08, "(higher is better ↑)", ha="center", va="bottom",
            fontsize=6.2, color="#555")

    # Reference marks
    y_ppl_ref = ppl_norm(PRETRAINED_PPL)
    ax.plot([-0.14, 0], [y_ppl_ref, y_ppl_ref], "--", color="#888", lw=1.0)
    ax.text(-0.15, y_ppl_ref, f"{PRETRAINED_PPL:.3f}\npretrained\nref.",
            ha="right", va="center", fontsize=5.5, color="#777", style="italic")

    y_auc_ref = auc_norm(UNLOCKED_AUROC)
    ax.plot([1, 1.14], [y_auc_ref, y_auc_ref], "--", color="#1b9e77", lw=1.0)
    ax.text(1.15, y_auc_ref, f"{UNLOCKED_AUROC:.3f}\nunlocked\nref.",
            ha="left", va="center", fontsize=5.5, color="#1b9e77", style="italic")

    # Draw each condition
    for (name, ppl, auroc, color, lw, group) in conds:
        y0 = ppl_norm(ppl)
        y1 = auc_norm(auroc)
        ax.plot([0, 1], [y0, y1], color=color, lw=lw, alpha=0.88, zorder=3)
        ax.scatter([0], [y0], color=color, s=28, zorder=7, edgecolors="white", lw=0.5)
        ax.scatter([1], [y1], color=color, s=28, zorder=7, edgecolors="white", lw=0.5)

    # Right-side labels: sorted by AUROC y-position with anti-overlap stagger
    raw_labels = [(auc_norm(c[2]), c[0], c[3]) for c in conds]
    raw_sorted  = sorted(raw_labels, key=lambda t: t[0])
    min_gap     = 0.055
    placed_y    = []
    for (raw_y, _, _) in raw_sorted:
        new_y = raw_y
        if placed_y:
            new_y = max(new_y, placed_y[-1] + min_gap)
        placed_y.append(new_y)
    for (raw_y, name, color), label_y in zip(raw_sorted, placed_y):
        ax.text(1.07, label_y, name, ha="left", va="center",
                fontsize=6.2, color=color)
        if abs(label_y - raw_y) > 0.015:
            ax.plot([1.04, 1.06], [raw_y, label_y], color=color, lw=0.5, alpha=0.5)

    ax.set_title(
        "Each line = one attack:  left = viral PPL,  right = mean AUROC\n"
        "Lines sloping up-right trade generative quality for virological capability",
        fontsize=7.5, pad=10,
    )
    plt.tight_layout(pad=0.8)
    _save("fig1_pareto")


# ==================================================================
# Figure 2 — (a) PPL bars  (b) AUROC bars  (representative conditions)
# ==================================================================
def fig1_scatter():
    """PPL–AUROC scatter with a broken x-axis.
    Left panel: main cluster (PPL 3.37–4.32). Right panel: Attack N outlier (PPL 5.60–6.10).
    """
    ATTACK_COL = "#34495e"   # uniform dark blue-gray for all attack points

    # (key, display_label, ppl, auroc, color, markersize)
    pts = [
        ("Pretrained", "Pretrained",          PRETRAINED_PPL,  PRETRAINED_AUROC,                  "#7f7f7f", 7),
        ("Unlocked",   "Unlocked FT",         UNLOCKED_PPL,    UNLOCKED_AUROC,                    "#1b9e77", 9),
        ("A",  "Naive (α=10⁴)",               3.7533, _mean(0.8640, 0.8178, 0.8617),              ATTACK_COL, 6),
        ("B",  "Naive η=10⁻⁶ (α=3×10⁴)",       3.8157, _mean(0.7972, 0.7883, 0.7851),              ATTACK_COL, 6),
        ("C",  "Naive η=10⁻⁵ (α=3×10⁴)",       3.7756, _mean(0.9045, 0.8471, 0.8939),              ATTACK_COL, 6),
        ("D",  "LoRA r=16",                    3.7290, _mean(0.8389, 0.7659, 0.7788),              ATTACK_COL, 6),
        ("E",  "Bypass (α=10⁴)",               4.0913, _mean(0.6883, 0.7353, 0.8453),              ATTACK_COL, 6),
        ("F",  "Bypass (α=3×10⁴)",             4.1228, _mean(0.7070, 0.7345, 0.8478),              ATTACK_COL, 6),
        ("G",  "SVD k=3 (α=10⁴)",             3.6670, _mean(0.8173, 0.7819, 0.8188),              ATTACK_COL, 6),
        ("H",  "SVD k=2 (α=3×10⁴)",           3.7450, _mean(0.8454, 0.7862, 0.8404),              ATTACK_COL, 6),
        ("I",  "SVD k=3 (α=3×10⁴)",           3.7051, _mean(0.8584, 0.8032, 0.8454),              ATTACK_COL, 6),
        ("J",  "SVD k=5 (α=3×10⁴)",           3.7603, _mean(0.8519, 0.8075, 0.8451),              ATTACK_COL, 6),
        ("K",  "Naive η=10⁻⁵ (α=10⁵)",        3.7984, _mean(0.8785, 0.8391, 0.8717),              ATTACK_COL, 6),
        ("L",  "Naive η=10⁻⁶ (α=10⁵)",        3.8760, _mean(0.8171, 0.7839, 0.8213),              ATTACK_COL, 6),
        ("M",  "Naive η=10⁻⁵ (α=3×10⁵)",      3.8099, _mean(0.8741, 0.8310, 0.8691),              ATTACK_COL, 6),
        ("N",  "Naive η=10⁻⁶ (α=3×10⁵)",      5.8607, _mean(0.8346, 0.8057, 0.8295),              "#c0392b", 6),
    ]

    YLIM   = (0.720, 0.930)
    XLIM_L = (3.37, 4.32)
    XLIM_R = (5.60, 6.10)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18 / 2.54, 12 / 2.54),
                                   gridspec_kw={"width_ratios": [5, 1], "wspace": 0.06})
    fig.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.12)

    for ax_ in (ax, ax2):
        ax_.set_ylim(*YLIM)
        ax_.grid(True, linewidth=0.3, color="#e8e8e8", zorder=0)
        ax_.set_axisbelow(True)
        ax_.tick_params(labelsize=7)
        ax_.axvline(PRETRAINED_PPL,  color="#888888", linestyle="--", linewidth=0.8, zorder=1)
        ax_.axhline(PRETRAINED_AUROC, color="#888888", linestyle=":",  linewidth=0.8, zorder=1)
        ax_.axhline(UNLOCKED_AUROC,  color="#1b9e77", linestyle="--", linewidth=0.8, zorder=1)

    # Shaded target region only in left panel (where it's visible)
    ax.fill_between([3.37, PRETRAINED_PPL], [UNLOCKED_AUROC, UNLOCKED_AUROC], [0.930, 0.930],
                    color="#1b9e77", alpha=0.07, zorder=0)

    ax.set_xlim(*XLIM_L)
    ax2.set_xlim(*XLIM_R)
    ax2.tick_params(labelleft=False)   # suppress duplicate y-tick labels on right panel
    ax2.spines["left"].set_linestyle((0, (4, 4)))   # dashed spine to mark the break

    # Broken-axis diagonal marks at the join
    d = 0.018
    for spine_ax, side in ((ax, "right"), (ax2, "left")):
        sign = +1 if side == "right" else -1
        xa = 1 if side == "right" else 0
        kw = dict(transform=spine_ax.transAxes, color="#555", clip_on=False, lw=1.0)
        spine_ax.plot((xa - sign * d, xa + sign * d), (-d, +d), **kw)
        spine_ax.plot((xa - sign * d, xa + sign * d), (1 - d, 1 + d), **kw)

    # Label offsets: (dx, dy, ha) in data coordinates of their respective panel
    label_offsets = {
        "Pretrained": (+0.007,  +0.004,  "left"),
        "Unlocked":   (+0.007,  +0.004,  "left"),
        "A":          (+0.007,  +0.012,  "left"),
        "B":          (+0.007,  -0.013,  "left"),
        "C":          (+0.007,  +0.004,  "left"),
        "D":          (+0.007,  -0.013,  "left"),
        "E":          (-0.007,  -0.013,  "right"),
        "F":          (+0.007,  +0.003,  "left"),
        "G":          (-0.007,  +0.004,  "right"),
        "H":          (+0.007,  +0.004,  "left"),
        "I":          (-0.007,  -0.013,  "right"),
        "J":          (+0.007,  -0.013,  "left"),
        "K":          (+0.007,  +0.004,  "left"),
        "L":          (+0.007,  -0.013,  "left"),
        "M":          (-0.007,  -0.013,  "right"),
        "N":          (-0.003,  +0.006,  "right"),
    }

    for key, full_lbl, ppl, auroc, color, ms in pts:
        target_ax = ax2 if key == "N" else ax
        target_ax.scatter(ppl, auroc, color=color, marker="o", s=ms ** 2,
                          edgecolors="white", linewidths=0.5, zorder=4)
        dx, dy, ha = label_offsets.get(key, (0.007, 0.003, "left"))
        is_ref = key in ("Pretrained", "Unlocked")
        fc = color if is_ref else "#1a1a1a"
        target_ax.text(ppl + dx, auroc + dy, full_lbl, fontsize=6, color=fc,
                       ha=ha, va="bottom", fontweight="bold" if is_ref else "normal")

    ax.set_xlabel("Held-out viral PPL", fontsize=8.5, labelpad=4)
    ax.set_ylabel("Mean HVUE AUROC", fontsize=8.5)
    # Shared x-label for right panel (smaller, positioned centrally)
    ax2.set_xlabel("PPL", fontsize=7, labelpad=4)

    _save("fig1_scatter")


# ==================================================================
# Figure 2 — Representative conditions bar chart
# ==================================================================
def fig2_main_bars():
    """Representative conditions: PPL and AUROC bars."""
    rows = [
        # (label, ppl, mean_auroc, color, hatch)
        ("Pretrained",          PRETRAINED_PPL,  PRETRAINED_AUROC,          "#7f7f7f", None),
        ("Unlocked",            UNLOCKED_PPL,    UNLOCKED_AUROC,            "#1b9e77", None),
        ("Naive\nη=10⁻⁶",      3.8157, _mean(0.7972, 0.7883, 0.7851),     "#d62728", None),
        ("Naive\nη=10⁻⁵",      3.7756, _mean(0.9045, 0.8471, 0.8939),     "#b22222", None),
        ("LoRA",                3.7290, _mean(0.8389, 0.7659, 0.7788),     "#ff7f0e", None),
        ("Bypass",              4.1228, _mean(0.7070, 0.7345, 0.8478),     "#7b2d8b", "////"),
        ("SVD k=2",             3.7450, _mean(0.8454, 0.7862, 0.8404),     "#4a90c2", None),
        ("SVD k=3",             3.7051, _mean(0.8584, 0.8032, 0.8454),     "#005f9e", None),
    ]
    labels  = [r[0] for r in rows]
    ppls    = [r[1] for r in rows]
    aurocs  = [r[2] for r in rows]
    colors  = [r[3] for r in rows]
    hatches = [r[4] for r in rows]

    fig = plt.figure(figsize=(22 / 2.54, 12 / 2.54))
    gs  = fig.add_gridspec(1, 2, wspace=0.46)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(rows))

    # ---- (a): PPL ----------------------------------------
    bars1 = ax1.bar(x, ppls, color=colors, edgecolor="black", linewidth=0.5)
    for bar, hatch in zip(bars1, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax1.axhline(PRETRAINED_PPL, color="#7f7f7f", linestyle="--", linewidth=0.9, zorder=0,
                label="Pretrained")
    ax1.axhline(UNLOCKED_PPL,   color="#1b9e77", linestyle="--", linewidth=0.9, zorder=0,
                label="Unlocked")
    ax1.set_ylabel("Held-out viral PPL")
    ax1.set_ylim(3.25, 4.40)
    ax1.set_yticks([3.4, 3.6, 3.8, 4.0, 4.2])
    ax1.set_title("(a) Perplexity", fontsize=8.5, pad=4)
    ax1.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=6.2, loc="upper left", borderpad=0.4, handlelength=1.5)
    for xi, v in zip(x, ppls):
        ax1.text(xi, v + 0.014, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=5.8, color="#222")

    # ---- (b): AUROC ----------------------------------------
    bars2 = ax2.bar(x, aurocs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, hatch in zip(bars2, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax2.axhline(PRETRAINED_AUROC, color="#7f7f7f", linestyle="--", linewidth=0.9, zorder=0,
                label="Pretrained")
    ax2.axhline(UNLOCKED_AUROC,   color="#1b9e77", linestyle="--", linewidth=0.9, zorder=0,
                label="Unlocked")
    ax2.set_ylabel("Mean HVUE AUROC")
    ax2.set_ylim(0.68, 0.95)
    ax2.set_yticks([0.70, 0.74, 0.78, 0.82, 0.86, 0.90])
    ax2.set_title("(b) Mean AUROC", fontsize=8.5, pad=4)
    ax2.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#dddddd")
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=6.2, loc="upper right", borderpad=0.4, handlelength=1.5)
    for xi, v in zip(x, aurocs):
        ax2.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=5.8, color="#222")

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.0, rotation=35, ha="right",
                           multialignment="center")
        ax.tick_params(axis="x", length=0)

    fig.subplots_adjust(bottom=0.22)
    plt.tight_layout(pad=1.2)
    _save("fig2_main_bars")


# ==================================================================
# Figure 3 — SVD-chain k and α ablation (all complete)
# ==================================================================
def fig3_kablation():
    fig, axes = plt.subplots(1, 2, figsize=(17 / 2.54, 8 / 2.54), sharey=False)
    fig.subplots_adjust(wspace=0.42)

    # Complete data: (k_label, alpha_label, ppl, mean_auroc, color)
    data = [
        ("k=2", "α=3×10⁴", 3.7450, _mean(0.8454, 0.7862, 0.8404), "#4a90c2"),
        ("k=3", "α=3×10⁴", 3.7051, _mean(0.8584, 0.8032, 0.8454), "#005f9e"),
        ("k=3", "α=10⁴",   3.6670, _mean(0.8173, 0.7819, 0.8188), "#aec7e8"),
        ("k=5", "α=3×10⁴", 3.7603, _mean(0.8519, 0.8075, 0.8451), "#003a6b"),
    ]

    all_labels = [f"{d[0]}\n{d[1]}" for d in data]
    n = len(all_labels)
    x = np.arange(n)

    ppls_all   = [d[2] for d in data]
    aurocs_all = [d[3] for d in data]
    colors_all = [d[4] for d in data]

    ax_ppl, ax_auc = axes
    for ax_ in (ax_ppl, ax_auc):
        ax_.grid(True, axis="y", linestyle="-", linewidth=0.35, color="#e0e0e0", zorder=0)
        ax_.set_axisbelow(True)

    # --- (a) PPL subplot ---
    ax_ppl.bar(x, ppls_all, color=colors_all,
               edgecolor="black", linewidth=0.5, width=0.55)
    ax_ppl.axhline(PRETRAINED_PPL, color="#7f7f7f", linestyle="--", linewidth=0.9, zorder=1,
                   label=f"Pretrained ({PRETRAINED_PPL:.3f})")
    ax_ppl.axhline(UNLOCKED_PPL,   color="#1b9e77", linestyle="--", linewidth=0.9, zorder=1,
                   label=f"Unlocked ({UNLOCKED_PPL:.3f})")
    ax_ppl.set_ylabel("Held-out viral PPL")
    ax_ppl.set_ylim(3.48, 4.02)
    ax_ppl.set_yticks([3.55, 3.65, 3.75, 3.85, 3.95])
    ax_ppl.set_title("(a) PPL vs chain length k", fontsize=8.5, pad=4)
    ax_ppl.legend(fontsize=6.0, loc="upper left", borderpad=0.4)
    for xi, v in zip(x, ppls_all):
        ax_ppl.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=6.0)

    # --- (b) AUROC subplot ---
    ax_auc.bar(x, aurocs_all, color=colors_all,
               edgecolor="black", linewidth=0.5, width=0.55)
    ax_auc.axhline(PRETRAINED_AUROC, color="#7f7f7f", linestyle="--", linewidth=0.9, zorder=1,
                   label=f"Pretrained ({PRETRAINED_AUROC:.3f})")
    ax_auc.axhline(UNLOCKED_AUROC,   color="#1b9e77", linestyle="--", linewidth=0.9, zorder=1,
                   label=f"Unlocked ({UNLOCKED_AUROC:.3f})")
    ax_auc.set_ylabel("Mean HVUE AUROC")
    ax_auc.set_ylim(0.77, 0.93)
    ax_auc.set_yticks([0.78, 0.81, 0.84, 0.87, 0.90])
    ax_auc.set_title("(b) AUROC vs chain length k", fontsize=8.5, pad=4)
    ax_auc.legend(fontsize=6.0, loc="upper right", borderpad=0.4)
    for xi, v in zip(x, aurocs_all):
        ax_auc.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=6.0)

    for ax_ in (ax_ppl, ax_auc):
        ax_.set_xticks(x)
        ax_.set_xticklabels(all_labels, fontsize=7.0, multialignment="center")
        ax_.tick_params(axis="x", length=0)

    plt.tight_layout(pad=1.0)
    _save("fig3_kablation")


# ==================================================================
# Figure S1 — Training curves (simplified single panel)
# ==================================================================
def _load_metrics(run_name, max_step=None, clip_val=None):
    path = os.path.join(RESULTS_DIR, run_name, "metrics.csv")
    steps, val = [], []
    if not os.path.exists(path):
        return steps, val
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
    """Simplified single-panel training loss curves for key configurations."""
    fig, ax = plt.subplots(figsize=(16 / 2.54, 8 / 2.54))
    ax.grid(True, linestyle="-", linewidth=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)

    runs = [
        ("ft_unlocked_25k_v2_unlocked",     "Unlocked FT",              "#1b9e77", "-",   25000),
        ("ft_locked_a30k_lr1e6_25k_locked", "Naive locked (α=3×10⁴)",   "#d62728", "-",   25000),
        ("ft_theorem8_a30k_k1_25k_locked",  "SVD-chain k=1 (α=3×10⁴)", "#1f77b4", "-",   25000),
        ("ft_theorem8_a30k_k3_25k_locked",  "SVD-chain k=3 (α=3×10⁴)", "#005f9e", "--",  25000),
        ("ft_bypass_a10k_25k_v2_locked",    "B-injection bypass",        "#9467bd", "-.",  25000),
        ("ft_lora_a10k_25k_locked",         "LoRA r=16",                "#ff7f0e", ":",   25000),
    ]
    for run, label, color, ls, mx in runs:
        steps, val = _load_metrics(run, max_step=mx, clip_val=1.55)
        if steps:
            ax.plot(steps, val, color=color, linestyle=ls, linewidth=1.3,
                    marker="o", markersize=1.5, label=label, zorder=3)

    pretrained_loss = 1.3161
    ax.axhline(pretrained_loss, color="#555", linestyle="--", linewidth=0.7, zorder=1)
    ax.text(28000, pretrained_loss + 0.004, "pretrained baseline",
            fontsize=6.5, color="#555", ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(80, 30000)
    ax.set_ylim(1.10, 1.52)
    ax.set_xlabel("Optimizer step (log scale)", fontsize=8)
    ax.set_ylabel("Validation loss (nats / token)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, handletextpad=0.4,
              borderpad=0.5, labelspacing=0.3, ncol=1)
    ax.set_title("Fine-tuning validation loss: key attack configurations",
                 fontsize=8.5, pad=5)

    plt.tight_layout(pad=1.0)
    _save("fig_s1_training_curves")


if __name__ == "__main__":
    fig1_scatter()
    fig2_main_bars()
    fig3_kablation()
    fig_s1()
    print("All figures written.")

