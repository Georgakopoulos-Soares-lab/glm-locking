"""Experiment A: Functional gene-content analysis of generated sequences.

Uses geNomad's per-sequence features.tsv (already computed by rescore_genomad.py)
for per-sequence gene counts, viral marker frequencies, and hallmark presence.

IMPORTANT STRUCTURAL LIMITATION
---------------------------------
Generated sequences are ~513 nt (max_new_tokens=512). Canonical viral hallmark
genes (RdRp, capsid protein, terminase, integrase) are 300–1000+ amino acids
→ 900–3000+ nt. At 513 nt you cannot fit an identifiable viral protein domain.
Result: 0/100 sequences in ALL conditions contain a viral hallmark gene.
This is a sequence-length artifact, NOT a biology finding.

What IS measurable at 513 nt
-----------------------------
* n_genes          : total ORF count (mean ~1 per sequence; very weak signal)
* v_marker_freq    : fraction of ORFs hitting geNomad viral markers (mostly 0)
* coding_density   : fraction of bases in ORFs (already in the main CSV)

We report bootstrap comparisons on these weak metrics honestly, flag the
limitation, and recommend re-running generation with max_new_tokens=4096.

For a definitive functional-coherence finding, run:
    python scripts/generative_comparison.py --max_new_tokens 4096
(This re-generates sequences; GPU required. Then re-run rescore_genomad.py.)
"""
from __future__ import annotations
import os, sys, csv
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENOMAD_DIR = "/tmp/genomad_rescore"
GEN_CSV = f"{REPO}/experiments/exp3_virobench/generative/generational_functional_scores.csv"
OUT_CSV = f"{REPO}/experiments/exp3_virobench/generative/gene_content_analysis.csv"
OUT_PAIRS = f"{REPO}/experiments/exp3_virobench/generative/gene_content_pairwise.csv"

MODELS = ["pretrained", "locked_no_ft", "unlocked_ft", "M_a300k"]
DISPLAY = {
    "pretrained":   "Pretrained",
    "locked_no_ft": "Locked-no-FT",
    "unlocked_ft":  "Unlocked-FT",
    "M_a300k":      "M (α=3×10⁵)",
}
N_BOOT = 10000
SEED = 0


# ── Load geNomad features per sequence ───────────────────────────────────────

def load_features(model: str) -> pd.DataFrame:
    feat_path = f"{GENOMAD_DIR}/{model}/{model}_marker_classification/{model}_features.tsv"
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"features.tsv not found: {feat_path}")
    feat = pd.read_csv(feat_path, sep="\t")
    # seq_name = {model}_{idx}
    feat["seq_idx"] = feat["seq_name"].str.rsplit("_", n=1).str[-1].astype(int)
    feat["model"] = model
    return feat


def load_all_features() -> dict[str, pd.DataFrame]:
    data = {}
    for m in MODELS:
        try:
            data[m] = load_features(m).set_index("seq_idx").sort_index()
        except FileNotFoundError as e:
            print(f"[warn] {e}")
    return data


# ── Bootstrap helpers ─────────────────────────────────────────────────────────

def boot_mean_diff(a: np.ndarray, b: np.ndarray, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap (same seq_idx) of mean(a) − mean(b)."""
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(a[idx].mean() - b[idx].mean())
    d = np.array(diffs)
    return float(a[:n].mean() - b[:n].mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ── Per-model summary ─────────────────────────────────────────────────────────

def summarise(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        if m not in data:
            continue
        f = data[m]
        n = len(f)
        # hallmark: any virus hallmark gene in this sequence
        vhall_any = (f["n_virus_hallmarks"] > 0).sum()
        any_viral_marker = (f["v_marker_freq"] > 0).sum()
        rows.append(dict(
            model=m,
            display=DISPLAY[m],
            n=n,
            seq_len_max_nt=513,  # hardcoded from generation config
            mean_n_genes=round(f["n_genes"].mean(), 3),
            std_n_genes=round(f["n_genes"].std(), 3),
            mean_v_marker_freq=round(f["v_marker_freq"].mean(), 4),
            mean_coding_density_genomad=round(f["coding_density"].mean(), 4),
            seqs_with_viral_hallmark=int(vhall_any),
            seqs_with_any_viral_marker=int(any_viral_marker),
            pct_viral_hallmark=round(100 * vhall_any / n, 1),
            pct_any_viral_marker=round(100 * any_viral_marker / n, 1),
        ))
    return pd.DataFrame(rows)


# ── Pairwise bootstrap comparisons ────────────────────────────────────────────

def pairwise_comparisons(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pair_defs = [
        ("path_artifact", "locked_no_ft", "pretrained"),
        ("ft_effect",     "unlocked_ft",  "pretrained"),
        ("lock_effect",   "M_a300k",      "unlocked_ft"),
    ]
    metrics = ["n_genes", "v_marker_freq", "coding_density"]
    rows = []
    for label, a_name, b_name in pair_defs:
        if a_name not in data or b_name not in data:
            continue
        fa = data[a_name]
        fb = data[b_name]
        # align by seq_idx
        idx = fa.index.intersection(fb.index)
        for metric in metrics:
            va = fa.loc[idx, metric].values.astype(float)
            vb = fb.loc[idx, metric].values.astype(float)
            delta, lo, hi = boot_mean_diff(va, vb)
            sig = "sig" if (lo > 0 or hi < 0) else "ns"
            rows.append(dict(
                comparison=label, a=a_name, b=b_name,
                metric=metric,
                delta=round(delta, 5),
                ci_lo=round(lo, 5),
                ci_hi=round(hi, 5),
                significant=sig,
                n_pairs=len(idx),
            ))
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Experiment A: Gene-content analysis of generated sequences")
    print("=" * 65)
    print()
    print("STRUCTURAL LIMITATION:")
    print("  Generated sequences are ~513 nt (max_new_tokens=512).")
    print("  Viral hallmark genes (RdRp, capsid, terminase) are 900-3000+ nt.")
    print("  => 0/100 sequences in ANY condition contain a viral hallmark.")
    print("  => This is a LENGTH ARTIFACT, not biology.")
    print("  => For definitive gene-content analysis, re-run generation with")
    print("     max_new_tokens=4096, then re-run rescore_genomad.py.")
    print()
    print("Available weak metrics (best-effort at 513 nt):")
    print("  n_genes         (ORF count; mean ~1 per seq)")
    print("  v_marker_freq   (viral marker fraction; 0 in unlocked/M)")
    print("  coding_density  (geNomad's estimate; see also Pyrodigal values)")
    print()

    data = load_all_features()
    if not data:
        print("[error] No geNomad features found in", GENOMAD_DIR)
        return

    summary = summarise(data)
    print("=== Per-model summary ===")
    print(summary.to_string(index=False))
    print()

    pairs = pairwise_comparisons(data)
    print("=== Pairwise bootstrap comparisons (n=10000) ===")
    print(pairs.to_string(index=False))
    print()

    # Check for hallmark = 0 everywhere — report the key null
    total_hallmark = sum(data[m]["n_virus_hallmarks"].sum() for m in data)
    if total_hallmark == 0:
        print("KEY FINDING: n_virus_hallmarks = 0 in ALL 400 sequences (all conditions).")
        print("This confirms the structural limitation: 513 nt is too short for")
        print("hallmark gene identification. v_marker_freq signal is near-zero noise.")
        print()
        print("RECOMMENDATION: Re-generate with max_new_tokens=4096:")
        print("  CUDA_VISIBLE_DEVICES=6 python scripts/generative_comparison.py \\")
        print("      --max_new_tokens 4096 --n_sequences 100")
        print("  Then: python scripts/rescore_genomad.py")
        print("  Then: python scripts/generative_gene_content.py")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    pairs.to_csv(OUT_PAIRS, index=False)
    print(f"\n[done] wrote {OUT_CSV}")
    print(f"[done] wrote {OUT_PAIRS}")


if __name__ == "__main__":
    main()
