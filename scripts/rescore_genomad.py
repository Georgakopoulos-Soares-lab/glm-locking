#!/usr/bin/env python3
"""Re-score cached generated sequences with geNomad end-to-end (batch per model).

Fixes the four bugs from generative_comparison.score_genomad:
  1. length gate (>=1000) blocked 513 bp seqs  -> no gate, geNomad handles short seqs
  2. wrong subcommand `annotate`               -> `end-to-end` (full classification)
  3. mmseqs missing                            -> installed into env bin
  4. DB path off by one level                  -> /data/genomad_db/genomad_db

Reads per-sequence scores from <prefix>_aggregated_classification.tsv (scores ALL
input sequences, not only viral-passing ones) and merges virus_score back into the
existing generative CSV.
"""
import os
import subprocess

import pandas as pd

REPO = "/home/nvidia/glm-locking"
GEN_DIR = f"{REPO}/experiments/exp3_virobench/generative"
SEQS_DIR = f"{GEN_DIR}/sequences"
CSV = f"{GEN_DIR}/generational_functional_scores.csv"
GENOMAD = "/home/nvidia/miniconda3/envs/evo/bin/genomad"
GENOMAD_DB = "/data/genomad_db/genomad_db"
WORK = "/tmp/genomad_rescore"
MODELS = ["pretrained", "locked_no_ft", "unlocked_ft", "M_a300k"]

env = dict(os.environ)
env["PATH"] = "/home/nvidia/miniconda3/envs/evo/bin:" + env.get("PATH", "")


def run_model(model: str) -> dict:
    """Run geNomad end-to-end on one model's FASTA; return {seq_name: virus_score}."""
    fasta = f"{SEQS_DIR}/{model}.fasta"
    out_dir = f"{WORK}/{model}"
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        GENOMAD, "end-to-end", fasta, out_dir, GENOMAD_DB,
        "--quiet", "--cleanup", "--disable-find-proviruses",
    ]
    print(f"[{model}] running geNomad end-to-end on {fasta} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    if r.returncode != 0:
        print(f"[{model}] geNomad FAILED (rc={r.returncode}):\n{r.stderr[-1500:]}", flush=True)

    prefix = os.path.splitext(os.path.basename(fasta))[0]
    agg = f"{out_dir}/{prefix}_aggregated_classification/{prefix}_aggregated_classification.tsv"
    if not os.path.exists(agg):
        print(f"[{model}] missing aggregated output: {agg}", flush=True)
        return {}
    df = pd.read_csv(agg, sep="\t")
    scores = dict(zip(df["seq_name"], df["virus_score"]))
    print(f"[{model}] scored {len(scores)} seqs; mean virus_score="
          f"{df['virus_score'].mean():.4f}", flush=True)
    return scores


def main():
    os.makedirs(WORK, exist_ok=True)
    all_scores = {}
    for m in MODELS:
        all_scores.update(run_model(m))

    d = pd.read_csv(CSV)
    keys = d["model"] + "_" + d["seq_idx"].astype(str)
    d["genomad_score"] = keys.map(all_scores)
    d["genomad_error"] = d["genomad_score"].isna().map({True: "no_score", False: ""})
    d.to_csv(CSV, index=False)

    n_scored = d["genomad_score"].notna().sum()
    print(f"\nMerged geNomad virus_score into {CSV}: {n_scored}/{len(d)} scored")
    print("\nPer-model mean virus_score:")
    print(d.groupby("display")["genomad_score"].agg(["mean", "std", "count"]).round(4).to_string())


if __name__ == "__main__":
    main()
