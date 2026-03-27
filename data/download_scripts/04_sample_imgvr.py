#!/usr/bin/env python3
"""
Step 4: Sample ~3,000 prokaryotic phages from IMG/VR for the retain dataset.

Reads the IMG/VR sequence information TSV, filters for:
  - High-confidence sequences
  - Prokaryotic host (Bacteria or Archaea)
  - Length >= 8,000 bp
  - Excludes 19 known eukaryotic virus families
  - 1 representative per vOTU

Then extracts the sampled sequences from the nucleotide FASTA using seqtk.

Usage:
    python data/download_scripts/04_sample_imgvr.py

Environment variables:
    SCRATCH_DIR  - root scratch directory (default: /scratch/10906/arisk/evo_locking_data)
"""

import csv
import os
import random
import subprocess
import sys
from collections import defaultdict

# ===========================================================================
# Config
# ===========================================================================
SEED = 42
N_PHAGES = 3000
MIN_SEQ_LEN = 8000

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch/10906/arisk/evo_locking_data")
IMGVR_DIR = os.path.join(SCRATCH_DIR, "imgvr")

SEQ_TSV = os.path.join(IMGVR_DIR, "IMGVR_all_Sequence_information-high_confidence.tsv")
NUCLEOTIDE_FNA = os.path.join(IMGVR_DIR, "IMGVR_all_nucleotides-high_confidence.fna")

OUTPUT_DIR = os.path.join(SCRATCH_DIR, "retain_build")
OUTPUT_IDS = os.path.join(OUTPUT_DIR, "imgvr_sampled_ids.txt")
OUTPUT_FASTA = os.path.join(OUTPUT_DIR, "imgvr_sampled.fasta")
OUTPUT_TSV = os.path.join(OUTPUT_DIR, "imgvr_sampled_metadata.tsv")

# Eukaryotic virus families to exclude
EXCLUDE_FAMILIES = {
    "Adenoviridae", "Ascoviridae", "Asfarviridae", "Baculoviridae",
    "Herpesviridae", "Iridoviridae", "Marseilleviridae", "Mimiviridae",
    "Nimaviridae", "Nudiviridae", "Papillomaviridae", "Phycodnaviridae",
    "Pithoviridae", "Polyomaviridae", "Poxviridae", "Retroviridae",
    "Reoviridae", "Togaviridae", "Flaviviridae",
}


def main():
    rng = random.Random(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Check inputs ---
    if not os.path.exists(SEQ_TSV):
        print(f"ERROR: {SEQ_TSV} not found. Run 03_download_imgvr.sh first.")
        sys.exit(1)

    # --- Parse and filter metadata ---
    print(f"Reading {SEQ_TSV}...")
    by_votu: dict[str, list[dict]] = defaultdict(list)

    total = 0
    passed_len = 0
    passed_host = 0
    passed_family = 0

    with open(SEQ_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            total += 1

            seq_id = row.get("UVIG", "").strip()
            if not seq_id:
                continue

            # Length filter
            try:
                seq_len = int(row.get("Length", "0"))
            except (ValueError, TypeError):
                continue
            if seq_len < MIN_SEQ_LEN:
                continue
            passed_len += 1

            # Host filter: prokaryotic only
            host_tax = row.get("Host taxonomy prediction", "")
            if not host_tax:
                continue
            host_lower = host_tax.lower()
            if "bacteria" not in host_lower and "archaea" not in host_lower:
                continue
            passed_host += 1

            # Family exclusion
            virus_tax = row.get("Taxonomic classification", "")
            excluded = False
            if virus_tax:
                for fam in EXCLUDE_FAMILIES:
                    if fam.lower() in virus_tax.lower():
                        excluded = True
                        break
            if excluded:
                continue
            passed_family += 1

            votu = row.get("vOTU", seq_id)
            rec = {
                "id": seq_id,
                "length": seq_len,
                "host": host_tax,
                "taxonomy": virus_tax,
                "votu": votu,
            }
            by_votu[votu].append(rec)

    print(f"\nFiltering summary:")
    print(f"  Total rows:            {total:,}")
    print(f"  After length >= {MIN_SEQ_LEN}:  {passed_len:,}")
    print(f"  After host filter:     {passed_host:,}")
    print(f"  After family excl:     {passed_family:,}")
    print(f"  Unique vOTUs:          {len(by_votu):,}")

    # 1 rep per vOTU (longest sequence)
    votu_reps = []
    for votu, recs in by_votu.items():
        best = max(recs, key=lambda r: r["length"])
        votu_reps.append(best)

    print(f"  vOTU representatives:  {len(votu_reps):,}")

    if len(votu_reps) <= N_PHAGES:
        sampled = votu_reps
        print(f"  Using all {len(sampled)} reps (fewer than target {N_PHAGES})")
    else:
        sampled = rng.sample(votu_reps, N_PHAGES)
        print(f"  Sampled {len(sampled)} from {len(votu_reps)} reps")

    # --- Write sequence IDs ---
    with open(OUTPUT_IDS, "w") as f:
        for rec in sampled:
            f.write(rec["id"] + "\n")
    print(f"\nWrote {len(sampled)} IDs to {OUTPUT_IDS}")

    # --- Write metadata ---
    with open(OUTPUT_TSV, "w") as f:
        f.write("sequence_id\tlength\thost_taxonomy\tvirus_taxonomy\tvotu\n")
        for rec in sampled:
            f.write(f"{rec['id']}\t{rec['length']}\t{rec['host']}\t{rec['taxonomy']}\t{rec['votu']}\n")
    print(f"Wrote metadata to {OUTPUT_TSV}")

    # --- Extract sequences with seqtk ---
    if not os.path.exists(NUCLEOTIDE_FNA):
        # Try .gz version
        fna_gz = NUCLEOTIDE_FNA + ".gz"
        if os.path.exists(fna_gz):
            nucleotide_path = fna_gz
        else:
            print(f"\nWARNING: Nucleotide FASTA not found at {NUCLEOTIDE_FNA}")
            print(f"         or {fna_gz}")
            print(f"         Run 03_download_imgvr.sh first.")
            print(f"         IDs have been saved — you can extract later with:")
            print(f"         seqtk subseq <fasta> {OUTPUT_IDS} > {OUTPUT_FASTA}")
            return
    else:
        nucleotide_path = NUCLEOTIDE_FNA

    # Check seqtk
    if subprocess.run(["which", "seqtk"], capture_output=True).returncode != 0:
        print("\nWARNING: seqtk not found. Install with: conda install -c bioconda seqtk")
        print(f"Then run: seqtk subseq {nucleotide_path} {OUTPUT_IDS} > {OUTPUT_FASTA}")
        return

    print(f"\nExtracting sequences from {nucleotide_path}...")
    with open(OUTPUT_FASTA, "w") as fout:
        result = subprocess.run(
            ["seqtk", "subseq", nucleotide_path, OUTPUT_IDS],
            stdout=fout,
            stderr=subprocess.PIPE,
            text=True,
        )
    if result.returncode != 0:
        print(f"ERROR: seqtk failed: {result.stderr}")
        sys.exit(1)

    # Count extracted
    n_extracted = 0
    with open(OUTPUT_FASTA) as f:
        for line in f:
            if line.startswith(">"):
                n_extracted += 1

    print(f"Extracted {n_extracted} sequences to {OUTPUT_FASTA}")
    print(f"Size: {os.path.getsize(OUTPUT_FASTA) / 1e6:.1f} MB")

    if n_extracted < len(sampled):
        print(f"WARNING: Expected {len(sampled)}, got {n_extracted}. "
              f"Some IDs may not match the FASTA headers.")

    # Length stats
    lengths = [r["length"] for r in sampled]
    lengths.sort()
    print(f"\nLength stats:")
    print(f"  Min:    {lengths[0]:,} bp")
    print(f"  Max:    {lengths[-1]:,} bp")
    print(f"  Median: {lengths[len(lengths)//2]:,} bp")
    print(f"  Total:  {sum(lengths):,} bp ({sum(lengths)/1e6:.1f} Mb)")


if __name__ == "__main__":
    main()
