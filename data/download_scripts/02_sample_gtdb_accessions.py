#!/usr/bin/env python3
"""
Step 2: Sample GTDB accessions for the retain dataset.

Reads taxonomy TSVs, samples 5,000 bacteria + 500 archaea
proportionally across phyla, writes an accession list for NCBI download.

Usage:
    python data/download_scripts/02_sample_gtdb_accessions.py
"""

import os
import random
import sys
from collections import Counter, defaultdict

SEED = 42
N_BACTERIA = 5000
N_ARCHAEA = 500

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch/10906/arisk/evo_locking_data")
GTDB_DIR = os.path.join(SCRATCH_DIR, "gtdb")

BAC_TAXONOMY = os.path.join(GTDB_DIR, "bac120_taxonomy_r220.tsv")
ARC_TAXONOMY = os.path.join(GTDB_DIR, "ar53_taxonomy_r220.tsv")

OUTPUT_ACCESSIONS = os.path.join(GTDB_DIR, "sampled_accessions.txt")
OUTPUT_TSV = os.path.join(GTDB_DIR, "sampled_metadata.tsv")


def parse_taxonomy(path: str) -> dict:
    ranks = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            acc, tax_str = parts
            levels = tax_str.split(";")
            rec = {rank: (lv.split("__", 1)[1] if "__" in lv else lv)
                   for rank, lv in zip(ranks, levels)}
            rec["gtdb_accession"] = acc
            # RS_GCF_000006945.2 -> GCF_000006945.2
            rec["ncbi_accession"] = acc.split("_", 1)[1] if acc.startswith(("RS_", "GB_")) else acc
            records[acc] = rec
    return records


def sample_across_phyla(records: dict, n_target: int, rng: random.Random) -> list:
    by_phylum = defaultdict(list)
    for rec in records.values():
        if rec.get("phylum"):
            by_phylum[rec["phylum"]].append(rec)

    total = sum(len(v) for v in by_phylum.values())
    sorted_phyla = sorted(by_phylum.keys(), key=lambda p: len(by_phylum[p]), reverse=True)

    # Proportional allocation, minimum 1 per phylum
    alloc = {p: max(1, int(len(by_phylum[p]) / total * n_target)) for p in by_phylum}
    current = sum(alloc.values())

    for p in sorted_phyla:
        if current >= n_target:
            break
        add = min(n_target - current, len(by_phylum[p]) - alloc[p])
        if add > 0:
            alloc[p] += add
            current += add

    sampled = []
    for p, recs in by_phylum.items():
        n = min(alloc.get(p, 0), len(recs))
        sampled.extend(rng.sample(recs, n))
    return sampled


def main():
    for path in [BAC_TAXONOMY, ARC_TAXONOMY]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run 01_download_gtdb_metadata.sh first.")
            sys.exit(1)

    print("Parsing bacterial taxonomy...")
    bac = parse_taxonomy(BAC_TAXONOMY)
    print(f"  {len(bac):,} species reps")

    print("Parsing archaeal taxonomy...")
    arc = parse_taxonomy(ARC_TAXONOMY)
    print(f"  {len(arc):,} species reps")

    print(f"\nSampling {N_BACTERIA} bacteria...")
    bac_sampled = sample_across_phyla(bac, N_BACTERIA, random.Random(SEED))
    print(f"Sampling {N_ARCHAEA} archaea...")
    arc_sampled = sample_across_phyla(arc, N_ARCHAEA, random.Random(SEED + 1))

    all_sampled = bac_sampled + arc_sampled
    print(f"Total: {len(all_sampled)}")

    bac_phyla = Counter(r["phylum"] for r in bac_sampled)
    print("\nBacterial phyla (top 10):")
    for p, c in bac_phyla.most_common(10):
        print(f"  {p:40s} {c:4d}")
    print(f"  ({len(bac_phyla)} phyla total)")

    # Write NCBI accession list (one per line, for datasets CLI)
    with open(OUTPUT_ACCESSIONS, "w") as f:
        for rec in all_sampled:
            f.write(rec["ncbi_accession"] + "\n")
    print(f"\nWrote {len(all_sampled)} accessions to {OUTPUT_ACCESSIONS}")

    # Write metadata TSV
    with open(OUTPUT_TSV, "w") as f:
        f.write("ncbi_accession\tgtdb_accession\tdomain\tphylum\tclass\torder\tfamily\tgenus\tspecies\n")
        for rec in all_sampled:
            f.write("\t".join([rec["ncbi_accession"], rec["gtdb_accession"],
                               rec["domain"], rec["phylum"], rec["class"],
                               rec["order"], rec["family"], rec["genus"], rec["species"]]) + "\n")
    print(f"Wrote metadata to {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
