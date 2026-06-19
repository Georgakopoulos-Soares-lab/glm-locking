"""Merge per-GPU LoRA result CSVs into a single summary table."""
import os
import pandas as pd
import glob

OUTDIR = "results/hvue_lora"
MERGED = "results/hvue_lora_all.csv"
TASKS = ["Host_Tropism", "Pathogenecity", "Transmissibility"]
CKPTS = ["pretrained", "locked_no_ft", "unlocked_ft", "M_a300k"]
KMER = {"Host_Tropism": 0.9030, "Pathogenecity": None, "Transmissibility": None}

dfs = []
for f in sorted(glob.glob(f"{OUTDIR}/*.csv")):
    try:
        df = pd.read_csv(f)
        if len(df):
            dfs.append(df)
            print(f"  loaded {f}: {len(df)} rows")
    except Exception as e:
        print(f"  skip {f}: {e}")

if not dfs:
    print("No CSVs found yet.")
else:
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(MERGED, index=False)
    print(f"\nMerged {len(all_df)} rows → {MERGED}")

    print("\n=== Best-LR summary per task × ckpt ===")
    best = all_df.loc[all_df.groupby(["task","ckpt"])["best_val_auroc"].idxmax()]
    for task in TASKS:
        t = best[best.task == task]
        if len(t):
            kmer = t["kmer_auroc"].iloc[0] if "kmer_auroc" in t.columns else "?"
            print(f"\n[{task}]  k-mer AUROC = {kmer}")
            cols = [c for c in ["ckpt","lr","best_val_auroc","residual",
                                 "n_lora_layers","is_locked"] if c in t.columns]
            print(t[cols].sort_values("ckpt").to_string(index=False))
