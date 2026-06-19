"""Gene-content analysis on the pre-collapse clean region of generated sequences.

Key argument:
  unlocked_ft generates clean viral-like sequence up to ~1968 nt (min onset),
  then collapses. M_a300k collapses at ~880 nt (min onset). So unlocked_ft
  produces 2× the coherent sequence M produces.

  This script:
    1. Extracts the safe pre-collapse window for each model:
         unlocked_ft : positions 0–1920 (below 1968 min onset, ACGT-only)
         M_a300k     : positions 0–864  (below 880 min onset, ACGT-only)
         pretrained  : positions 0–1920 (never collapses — same window as unlocked_ft)
         locked_no_ft: positions 0–1920 (same)
    2. Strips any residual N (shouldn't be any below the onset floor) and skips
       sequences too short for Prodigal (< 100 nt).
    3. Runs pyrodigal (meta mode = best for short/unknown sequences) on each window.
    4. Reports per-model: n_genes, coding_density, gene_length_mean, +/- strand ratio.
    5. Runs geNomad end-to-end on the same windows to get virus-score.
    6. Reports comparison: unlocked_ft vs M_a300k on the clean-region claim.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/gene_content_clean_region.py
"""
from __future__ import annotations
import os, sys, subprocess, shutil, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEQ_DIR  = f"{REPO}/experiments/exp3_virobench/generative_4096/sequences"
OUT_DIR  = f"{REPO}/experiments/exp3_virobench/gene_content_clean"
GENOMAD_DB = "/data/genomad_db/genomad_db"
GENOMAD_BIN = "/home/nvidia/miniconda3/envs/evo/bin/genomad"
# Ensure mmseqs (in evo env) is on PATH for geNomad
os.environ["PATH"] = "/home/nvidia/miniconda3/envs/evo/bin:" + os.environ.get("PATH", "")

# Pre-collapse safe windows (both endpoints below all-sequence min onset)
SAFE_WINDOWS = {
    "pretrained":   (0, 1920),  # never collapses; same window as unlocked_ft
    "locked_no_ft": (0, 1920),  # never collapses
    "unlocked_ft":  (0, 1920),  # min onset = 1968; stays clean to 1920
    "M_a300k":      (0, 864),   # min onset = 880; stays clean to 864
}

models = list(SAFE_WINDOWS.keys())


def load_seqs(model: str) -> list[str]:
    seqs, cur = [], []
    with open(f"{SEQ_DIR}/{model}.fasta") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur: seqs.append("".join(cur)); cur = []
            else: cur.append(line.strip())
        if cur: seqs.append("".join(cur))
    return seqs


def extract_window(seqs, lo, hi) -> list[str]:
    """Extract [lo:hi] and keep only ACGT."""
    out = []
    for s in seqs:
        chunk = "".join(c for c in s[lo:hi] if c in "ACGT")
        out.append(chunk)
    return out


def write_fasta(seqs, path, prefix):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            if len(s) >= 100:
                fh.write(f">{prefix}_{i}\n{s}\n")


def run_pyrodigal(seqs, model_name):
    """Run pyrodigal in meta mode and return per-sequence gene stats."""
    import pyrodigal
    stats = []
    orf_finder = pyrodigal.GeneFinder(meta=True)
    for s in seqs:
        if len(s) < 100:
            stats.append(dict(n_genes=0, coding_nt=0, seq_len=len(s)))
            continue
        genes = orf_finder.find_genes(s)
        coding_nt = sum(abs(g.end - g.begin) + 1 for g in genes)
        stats.append(dict(
            n_genes=len(genes),
            coding_nt=coding_nt,
            seq_len=len(s),
            gene_lens=[abs(g.end - g.begin) + 1 for g in genes],
            strands=[g.strand for g in genes],
        ))
    return stats


def summarize_prodigal(stats, label):
    n_seqs    = len(stats)
    n_genes   = [s["n_genes"] for s in stats]
    seq_lens  = [s["seq_len"]  for s in stats]
    coding_nt = [s["coding_nt"] for s in stats]
    density   = [c / max(l, 1) for c, l in zip(coding_nt, seq_lens)]
    all_lens  = [l for s in stats for l in s.get("gene_lens", [])]
    print(f"\n  [{label}]")
    print(f"    n_seqs       = {n_seqs}")
    print(f"    window_len   = {np.mean(seq_lens):.0f} nt (mean usable ACGT after trim)")
    print(f"    n_genes      = {np.mean(n_genes):.2f} ± {np.std(n_genes):.2f} per seq")
    print(f"    coding_den   = {np.mean(density):.4f} ± {np.std(density):.4f}")
    if all_lens:
        print(f"    gene_len_mean= {np.mean(all_lens):.0f} nt  median={np.median(all_lens):.0f} nt")
    return dict(
        label=label,
        n_seqs=n_seqs,
        window_len_mean=round(np.mean(seq_lens), 1),
        genes_mean=round(np.mean(n_genes), 3),
        genes_std=round(np.std(n_genes),  3),
        coding_density_mean=round(np.mean(density), 4),
        coding_density_std=round(np.std(density),  4),
        gene_len_mean=round(np.mean(all_lens), 1) if all_lens else 0,
    )


def run_genomad(fasta_path, out_dir):
    """Run geNomad end-to-end on fasta, return path to summary TSV."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        GENOMAD_BIN, "end-to-end",
        "--cleanup",
        "--splits", "4",
        "--disable-find-proviruses",   # skip aragorn dependency (no proviruses in short seqs)
        "--skip-trna-identification",  # skip aragorn dependency
        fasta_path,
        out_dir,
        GENOMAD_DB,
    ]
    print(f"    [genomad] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [genomad ERROR] {result.stderr[-500:]}", flush=True)
        return None
    # Find summary tsv
    for root, dirs, files in os.walk(out_dir):
        for fn in files:
            if "summary.tsv" in fn and "virus" in fn:
                return os.path.join(root, fn)
    return None


def parse_genomad_summary(tsv_path) -> dict:
    """Parse geNomad virus_summary.tsv → {seq_id: virus_score}."""
    scores = {}
    if not tsv_path or not os.path.exists(tsv_path):
        return scores
    with open(tsv_path) as fh:
        header = fh.readline().strip().split("\t")
        name_col = header.index("seq_name") if "seq_name" in header else 0
        score_col = next((i for i, h in enumerate(header) if "score" in h.lower()), 1)
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) > score_col:
                try:
                    scores[parts[name_col]] = float(parts[score_col])
                except ValueError:
                    pass
    return scores


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 65)
    print("Gene-content analysis: pre-collapse clean region")
    print("=" * 65)

    all_prodigal = {}
    all_genomad  = {}

    for model in models:
        lo, hi = SAFE_WINDOWS[model]
        seqs = load_seqs(model)
        windows = extract_window(seqs, lo, hi)
        window_len = hi - lo
        print(f"\n{'='*50}")
        print(f"Model: {model}   window: [{lo}, {hi}) = {window_len} nt", flush=True)

        # ── pyrodigal ─────────────────────────────────────────────
        stats = run_pyrodigal(windows, model)
        row = summarize_prodigal(stats, model)
        all_prodigal[model] = row

        # ── Write FASTA for geNomad ────────────────────────────────
        fasta_out = f"{OUT_DIR}/{model}_clean.fasta"
        write_fasta(windows, fasta_out, model)
        n_seqs_written = sum(1 for s in windows if len(s) >= 100)
        print(f"    [fasta] {n_seqs_written} seqs written → {fasta_out}", flush=True)

        # ── geNomad ───────────────────────────────────────────────
        genomad_out = f"{OUT_DIR}/genomad_{model}"
        tsv = run_genomad(fasta_out, genomad_out)
        scores = parse_genomad_summary(tsv)
        # Any seq that geNomad classifies as virus gets a score; unclassified = 0
        all_scores = [scores.get(f"{model}_{i}", 0.0)
                      for i in range(len(windows)) if len(windows[i]) >= 100]
        n_classified = sum(1 for v in all_scores if v > 0)
        mean_score = float(np.mean(all_scores)) if all_scores else 0.0
        print(f"    [genomad] n_classified={n_classified}/{n_seqs_written}  "
              f"mean_score={mean_score:.4f}", flush=True)
        all_genomad[model] = dict(
            n_classified=n_classified,
            n_seqs=n_seqs_written,
            frac_classified=round(n_classified / max(n_seqs_written, 1), 4),
            mean_virus_score=round(mean_score, 4),
            all_scores=all_scores,
        )

    # ── Same-window normalization: compare all models at 864 nt ───
    print("\n")
    print("=" * 65)
    print("NORMALIZED COMPARISON: all models at same 864-nt window")
    print("=" * 65)
    norm_stats = {}
    for model in models:
        seqs = load_seqs(model)
        windows_864 = extract_window(seqs, 0, 864)
        stats_864 = run_pyrodigal(windows_864, model)
        row_864 = summarize_prodigal(stats_864, f"{model}@864")
        norm_stats[model] = row_864

    print(f"{'Model':<16} {'Genes/seq':>10} {'CodingDen':>11} {'GeneLen':>10}")
    print("-" * 51)
    for model in models:
        r = norm_stats[model]
        print(f"{model:<16} {r['genes_mean']:>10.3f} "
              f"{r['coding_density_mean']:>11.4f} {r['gene_len_mean']:>10.1f}")

    # ── Final comparison ──────────────────────────────────────────
    print("\n")
    print("=" * 65)
    print("SUMMARY: Clean-region gene content, per model")
    print("=" * 65)
    print(f"{'Model':<16} {'Window':>8} {'Genes/seq':>10} {'CodingDen':>11} "
          f"{'GeNomadFrac':>13} {'GeNomadScore':>14}")
    print("-" * 76)
    for model in models:
        lo, hi = SAFE_WINDOWS[model]
        p = all_prodigal[model]
        g = all_genomad[model]
        print(f"{model:<16} {lo}-{hi:>4} {p['genes_mean']:>10.3f} "
              f"{p['coding_density_mean']:>11.4f} "
              f"{g['frac_classified']:>13.4f} {g['mean_virus_score']:>14.4f}")

    print("\nKey claim test:")
    uf = all_prodigal["unlocked_ft"]
    ma = all_prodigal["M_a300k"]
    print(f"  unlocked_ft  clean region [{SAFE_WINDOWS['unlocked_ft'][0]}-"
          f"{SAFE_WINDOWS['unlocked_ft'][1]}]: {uf['genes_mean']:.2f} genes/seq, "
          f"coding={uf['coding_density_mean']:.4f}")
    print(f"  M_a300k clean region [{SAFE_WINDOWS['M_a300k'][0]}-"
          f"{SAFE_WINDOWS['M_a300k'][1]}]: {ma['genes_mean']:.2f} genes/seq, "
          f"coding={ma['coding_density_mean']:.4f}")
    print(f"  unlocked_ft  has {uf['genes_mean']/max(ma['genes_mean'],0.01):.1f}x "
          f"more genes and {(SAFE_WINDOWS['unlocked_ft'][1]-SAFE_WINDOWS['unlocked_ft'][0])}"
          f" vs {(SAFE_WINDOWS['M_a300k'][1]-SAFE_WINDOWS['M_a300k'][0])} nt clean window")

    guf = all_genomad["unlocked_ft"]
    gma = all_genomad["M_a300k"]
    print(f"  geNomad: unlocked_ft score={guf['mean_virus_score']:.4f} "
          f"({guf['frac_classified']:.1%} classified)  "
          f"M_a300k score={gma['mean_virus_score']:.4f} "
          f"({gma['frac_classified']:.1%} classified)")

    # Save CSV
    import csv
    csv_out = f"{OUT_DIR}/clean_region_summary.csv"
    with open(csv_out, "w", newline="") as fh:
        fn = ["model", "window_lo", "window_hi", "window_len",
              "genes_per_seq", "genes_std", "coding_density", "coding_density_std",
              "gene_len_mean", "genomad_frac", "genomad_score_mean"]
        w = csv.DictWriter(fh, fieldnames=fn)
        w.writeheader()
        for model in models:
            lo, hi = SAFE_WINDOWS[model]
            p = all_prodigal[model]
            g = all_genomad[model]
            w.writerow({
                "model": model, "window_lo": lo, "window_hi": hi,
                "window_len": hi-lo,
                "genes_per_seq": p["genes_mean"],
                "genes_std": p["genes_std"],
                "coding_density": p["coding_density_mean"],
                "coding_density_std": p["coding_density_std"],
                "gene_len_mean": p["gene_len_mean"],
                "genomad_frac": g["frac_classified"],
                "genomad_score_mean": g["mean_virus_score"],
            })
    print(f"\n[saved] {csv_out}")


if __name__ == "__main__":
    main()
