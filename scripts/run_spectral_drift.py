#!/usr/bin/env python3
"""Evo ΔW Spectral Drift Analysis.

For each of the 32 locked output projections in Evo-1-8k-base, measures where
fine-tuning weight updates concentrate in the pretrained SVD basis.

Mechanistic hypothesis:
  Virological capability lives in the top singular directions of W_pre.
  → Unlocked FT concentrates ΔW in these top-σ directions (recovers AUROC).
  → Naive locked FT produces diffuse, noise-level ΔW_eff (AUROC drops).
  → SVD-chain also concentrates ΔW in top-σ directions (explains AUROC recovery).

Usage:
  python scripts/run_spectral_drift.py [--n_dirs 256] [--out results/spectral_drift.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
PRETRAINED_PT = (
    "/home/nvidia/.cache/huggingface/hub/models--togethercomputer--evo-1-8k-base"
    "/snapshots/6d7baa4482172f2c451ca4b36c87d50c8359a134/pytorch_model.pt"
)
RESULTS_DIR = Path("/home/nvidia/evo-locking/results")

# Checkpoints to analyze: {name: (result_dir, label)}
CHECKPOINTS = {
    "unlocked_ft": (
        RESULTS_DIR / "ft_attack_specdef_alpha10k_topk3000_unlocked",
        "Unlocked FT (α=10k baseline)",
    ),
    "naive_locked": (
        RESULTS_DIR / "ft_attack_specdef_alpha10k_topk3000_locked",
        "Naive locked FT (α=10k)",
    ),
    "svdchain_k2": (
        RESULTS_DIR / "ft_attack_theorem8_paper_all_locked",
        "SVD-chain k=2 (Theorem 8, α=10k)",
    ),
}

# ── Key resolution ────────────────────────────────────────────────────────────
def _target_weight_keys(state_dict: dict) -> list[str]:
    """Return keys for the 32 Evo output projection weight tensors."""
    keys = []
    for i in range(32):
        for template in (
            f"blocks.{i}.out_filter_dense.weight",        # unlocked FT: flat
            f"blocks.{i}.out_filter_dense.linear.weight", # locked SpecDef wrapper
        ):
            if template in state_dict:
                keys.append(template)
                break
    for i in [8, 16, 24]:
        for template in (
            f"blocks.{i}.inner_mha_cls.out_proj.weight",
            f"blocks.{i}.inner_mha_cls.out_proj.linear.weight",
        ):
            if template in state_dict:
                keys.append(template)
                break
    return keys


# ── Effective weight extraction ───────────────────────────────────────────────
def _comp_key(base_key: str) -> str:
    """Given a .linear.weight or .weight key, return the .comp.weight sibling."""
    return re.sub(r"(linear\.)?weight$", "comp.weight", base_key)


def _factor_keys(base_key: str, state_dict: dict) -> list[str]:
    """Return ordered list of .factors.{i}.weight keys for a given base prefix."""
    prefix = re.sub(r"(\.linear)?\.weight$", "", base_key)
    found = []
    for i in range(20):
        fk = f"{prefix}.factors.{i}.weight"
        if fk in state_dict:
            found.append(fk)
        elif found:
            break
    return found


def _effective_weight(key: str, state_dict: dict) -> torch.Tensor:
    """Return the effective W_eff for a given target layer.

    Three formats:
      1. Unlocked FT: plain state_dict[key]  [4096, 4096]
      2. Naive locked (SpecDef): C @ W̃     where C = comp.weight, W̃ = linear.weight
      3. Theorem8 (SVD-chain): C @ L_k @ ... @ L_0  (factors fused with comp)
    """
    # Detect Theorem8 by presence of .factors.0.weight sibling
    fkeys = _factor_keys(key, state_dict)
    if fkeys:
        # Fuse: W_factors = L_{k-1} @ ... @ L_0
        w = state_dict[fkeys[0]].double()
        for fk in fkeys[1:]:
            w = state_dict[fk].double() @ w
        # Apply compensation: C is always present with Theorem8
        ck = _comp_key(key)
        if ck in state_dict:
            w = state_dict[ck].double() @ w
        return w.float()

    # Detect SpecDef by presence of .comp.weight sibling
    ck = _comp_key(key)
    if ck in state_dict and key in state_dict:
        W_tilde = state_dict[key].double()
        C = state_dict[ck].double()
        return (C @ W_tilde).float()

    # Plain weight (unlocked FT)
    return state_dict[key].float()


# ── SVD projection ────────────────────────────────────────────────────────────
def project_delta(
    W_pre: torch.Tensor, W_ft_eff: torch.Tensor, n_dirs: int
) -> dict:
    """Measure where ΔW_eff concentrates in the pretrained left-singular subspace.

    Primary metric: full-subspace projection
        subspace_frac(r) = ||U_r^T ΔW||_F^2 / ||ΔW||_F^2
    where U_r = top-r left singular vectors of W_pre.
    This captures all ΔW energy that maps into the top-r output directions of W_pre,
    whether or not it aligns with individual (u_i, v_i) pairs.

    enrichment(r) = subspace_frac(r) / (r / d)  — ratio vs uniform expectation.

    Returns dict with:
      - subspace_frac:  (n_dirs,) — frac of ΔW energy in top-r left singular subspace
      - enrichment:     (n_dirs,) — subspace_frac / (r/d), ratio vs uniform baseline
      - delta_norm: scalar     — ||ΔW||_F
      - sigma_max: scalar      — σ_1 of W_pre
      - top10_frac: scalar     — subspace_frac[9]
      - top10_enrich: scalar   — enrichment at top-10
    """
    W_pre = W_pre.float()
    W_ft_eff = W_ft_eff.float()
    dW = W_ft_eff - W_pre

    delta_norm = float(dW.norm())
    dW_norm_sq = float(dW.pow(2).sum())
    d_out = W_pre.shape[0]

    if dW_norm_sq < 1e-12:
        return {
            "subspace_frac": [0.0] * n_dirs,
            "enrichment": [0.0] * n_dirs,
            "delta_norm": delta_norm,
            "sigma_max": 0.0,
            "top10_frac": 0.0,
            "top10_enrich": 0.0,
        }

    # Full SVD of pretrained weight (thin, CPU)
    U, S, Vh = torch.linalg.svd(W_pre, full_matrices=False)
    sigma_max = float(S[0])

    # Compute subspace_frac(r) incrementally: add projection of (u_r^T ΔW) at each step
    # u_r^T dW  has shape [d_in]
    # ||u_r^T dW||^2 = energy absorbed by the r-th left singular direction
    subspace_frac = np.zeros(n_dirs)
    cumulative_energy = 0.0
    for r in range(n_dirs):
        u_r = U[:, r]                        # [d_out]
        row_energy = float((u_r @ dW).pow(2).sum())  # ||u_r^T ΔW||^2
        cumulative_energy += row_energy
        subspace_frac[r] = cumulative_energy / dW_norm_sq

    baseline = (np.arange(1, n_dirs + 1) / d_out)
    enrichment = subspace_frac / np.maximum(baseline, 1e-12)

    return {
        "subspace_frac": subspace_frac.tolist(),
        "enrichment": enrichment.tolist(),
        "delta_norm": delta_norm,
        "sigma_max": sigma_max,
        "top10_frac": float(subspace_frac[9]),
        "top10_enrich": float(enrichment[9]),
    }


# ── Per-checkpoint analysis ───────────────────────────────────────────────────
def analyze_checkpoint(
    name: str,
    ckpt_dir: Path,
    label: str,
    pretrained: dict,
    n_dirs: int,
) -> dict:
    print(f"\n── {name}: {label}")
    ckpt_path = ckpt_dir / "model_best.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] {ckpt_path} not found", file=sys.stderr)
        return None

    state_dict = torch.load(ckpt_path, map_location="cpu")
    print(f"  Loaded: {len(state_dict)} keys")

    # Identify which target keys exist in the FT checkpoint
    # For Theorem8, the key structure diverges — identify by comp/factors siblings
    pre_keys = _target_weight_keys(pretrained)
    ft_keys = _target_weight_keys(state_dict)
    # Theorem8 stores factors, not a flat weight — fall back to pretrained key patterns
    # and check for factor siblings
    use_pre_keys = pre_keys  # always project ΔW relative to these canonical key prefixes

    matched = []
    for pk in use_pre_keys:
        # Resolved pretrained weight
        W_pre = pretrained[pk].float()
        # For FT checkpoint: find corresponding key
        # Theorem8 may not have flat weight but has factors
        base_prefix = re.sub(r"\.weight$", "", pk)
        # Check if factors exist
        fkeys = _factor_keys(pk, state_dict)
        ck = _comp_key(pk)
        if fkeys:
            W_eff = _effective_weight(pk, state_dict)
            matched.append((pk, W_pre, W_eff))
        elif ck in state_dict:
            # SpecDef wrapper (locked naive FT uses .linear.weight)
            lk = pk.replace(".weight", ".linear.weight") if ".linear.weight" not in pk else pk
            if lk not in state_dict:
                lk = pk  # fallback
            if lk in state_dict:
                W_eff = _effective_weight(lk, state_dict)
                matched.append((pk, W_pre, W_eff))
        elif pk in state_dict:
            W_eff = state_dict[pk].float()
            matched.append((pk, W_pre, W_eff))
        else:
            print(f"  [WARN] key not found in FT checkpoint: {pk}")

    print(f"  Matched layers: {len(matched)} / {len(use_pre_keys)}")

    layers = []
    all_sf = []
    all_enrich = []
    for pk, W_pre, W_eff in matched:
        proj = project_delta(W_pre, W_eff, n_dirs)
        layers.append({"key": pk, **proj})
        all_sf.append(proj["subspace_frac"])
        all_enrich.append(proj["enrichment"])
        dn = proj["delta_norm"]
        t10 = proj["top10_frac"]
        e10 = proj["top10_enrich"]
        print(f"    {pk:55s}  ||ΔW||={dn:7.3f}  top10_frac={t10:.4f}  top10_enrich={e10:.2f}x")

    mean_sf     = np.stack(all_sf).mean(axis=0).tolist()     if all_sf     else []
    mean_enrich = np.stack(all_enrich).mean(axis=0).tolist() if all_enrich else []
    mean_delta_norm = float(np.mean([l["delta_norm"] for l in layers])) if layers else 0.0
    x = np.arange(1, n_dirs + 1)
    random_baseline = (x / 4096).tolist()

    print(f"  Mean ||ΔW||: {mean_delta_norm:.4f}")
    if mean_sf:
        e10_mean = mean_enrich[9] if len(mean_enrich) > 9 else float('nan')
        print(f"  Mean top10_frac={mean_sf[9]:.4f}  top10_enrich={e10_mean:.2f}x")

    return {
        "name": name,
        "label": label,
        "mean_delta_norm": mean_delta_norm,
        "mean_subspace_frac": mean_sf,
        "mean_enrichment": mean_enrich,
        "random_baseline": random_baseline,
        "layers": layers,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(all_results: list[dict], n_dirs: int, plot_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "unlocked_ft": "steelblue",
        "naive_locked": "tomato",
        "svdchain_k2": "mediumseagreen",
    }
    x = np.arange(1, n_dirs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: subspace_frac vs rank (cumulative fraction of ΔW in top-r left singular subspace)
    ax = axes[0]
    rb = np.array(all_results[0]["random_baseline"]) if all_results else x / 4096
    ax.plot(x, rb, "k--", lw=1.5, alpha=0.5, label="Uniform (k/d)")
    for r in all_results:
        sf = np.array(r["mean_subspace_frac"])
        if len(sf) == 0:
            continue
        ax.plot(x[:len(sf)], sf, color=colors.get(r["name"], "gray"),
                lw=2.2, label=r["label"])
    ax.set_xlabel("Top-r left singular directions of W_pre", fontsize=11)
    ax.set_ylabel("Fraction of ||\u0394W||\u00b2 in top-r subspace", fontsize=11)
    ax.set_title("ΔW subspace concentration (||U_r\u1d40 ΔW||\u00b2_F / ||ΔW||\u00b2_F)\n(mean over 32 output projections)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel B: enrichment ratio at key rank thresholds
    ax2 = axes[1]
    ks = [1, 5, 10, 25, 50, 100]
    width = 0.22
    xpos = np.arange(len(ks))
    for j, r in enumerate(all_results):
        en = np.array(r["mean_enrichment"])
        if len(en) == 0:
            continue
        vals = [float(en[min(k - 1, len(en) - 1)]) for k in ks]
        ax2.bar(
            xpos + j * width, vals, width,
            label=r["label"], color=colors.get(r["name"], "gray"), alpha=0.85,
        )
    ax2.axhline(1.0, color="k", ls="--", lw=1.5, alpha=0.5, label="Uniform baseline (1.0\u00d7)")
    ax2.set_xticks(xpos + width)
    ax2.set_xticklabels([f"top-{k}" for k in ks], fontsize=9)
    ax2.set_ylabel("Enrichment ratio (observed / uniform)", fontsize=11)
    ax2.set_title("ΔW spectral enrichment at W_pre top singular subspaces", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {plot_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_dirs", type=int, default=256,
                    help="Number of SVD directions to project onto (default: 256)")
    ap.add_argument("--out", default="results/spectral_drift.json",
                    help="Output JSON path")
    ap.add_argument("--plot", default="results/spectral_drift.png",
                    help="Output plot path")
    ap.add_argument("--ckpts", nargs="+", default=list(CHECKPOINTS.keys()),
                    help=f"Checkpoints to analyze. Options: {list(CHECKPOINTS.keys())}")
    args = ap.parse_args()

    print(f"Loading pretrained weights from:\n  {PRETRAINED_PT}")
    pretrained = torch.load(PRETRAINED_PT, map_location="cpu")
    pre_keys = _target_weight_keys(pretrained)
    print(f"Pretrained: {len(pre_keys)} target projection keys found")

    all_results = []
    for name in args.ckpts:
        if name not in CHECKPOINTS:
            print(f"Unknown checkpoint '{name}'. Options: {list(CHECKPOINTS.keys())}", file=sys.stderr)
            continue
        ckpt_dir, label = CHECKPOINTS[name]
        r = analyze_checkpoint(name, ckpt_dir, label, pretrained, args.n_dirs)
        if r is not None:
            all_results.append(r)

    # Print summary table
    print("\n" + "=" * 80)
    hdr = f"{'Condition':<40} {'mean ||dW||':>12}  {'top10_frac':>10}  {'top10_enrich':>12}"
    print(hdr)
    print("-" * 80)
    for r in all_results:
        sf  = r["mean_subspace_frac"]
        en  = r["mean_enrichment"]
        t10 = sf[9] if len(sf) > 9 else float("nan")
        e10 = en[9] if len(en) > 9 else float("nan")
        print(f"  {r['label']:<38} {r['mean_delta_norm']:>12.4f}  {t10:>10.4f}  {e10:>12.2f}x")
    mdash = "\u2014"
    print(f"  {'Uniform baseline':<38} {mdash:>12}  {10/4096:>10.4f}  {'1.00x':>12}")
    print("=" * 80)

    out_data = {
        "n_dirs": args.n_dirs,
        "results": [
            {k: v for k, v in r.items() if k != "layers"}  # strip per-layer data
            for r in all_results
        ],
        "results_with_layers": all_results,
    }
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved: {args.out}")

    if all_results:
        plot_results(all_results, args.n_dirs, args.plot)


if __name__ == "__main__":
    main()
