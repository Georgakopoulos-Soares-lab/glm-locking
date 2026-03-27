#!/usr/bin/env python3
"""
Step 4: Extract 1 random contig >= 10 kb per GTDB genome -> gtdb_sampled.fasta

Reads the .fna files downloaded in step 3 and picks one contig per genome.

Usage:
    python data/download_scripts/04_extract_gtdb_contigs.py
"""

import os
import random
import sys

SEED = 42
MIN_CONTIG_LEN = 10_000  # 10 kb

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch/10906/arisk/evo_locking_data")
GENOME_DIR = os.path.join(SCRATCH_DIR, "gtdb", "genomes")
OUTPUT_DIR = os.path.join(SCRATCH_DIR, "retain_build")
OUTPUT_FASTA = os.path.join(OUTPUT_DIR, "gtdb_sampled.fasta")


def parse_fasta(path: str) -> list:
    records = []
    header = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.upper())
        if header is not None:
            records.append((header, "".join(seq_parts)))
    return records


def clean_dna(seq: str) -> str:
    return "".join(c for c in seq.upper().replace("U", "T") if c in "ACGT")


def main():
    rng = random.Random(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fna_files = sorted(f for f in os.listdir(GENOME_DIR) if f.endswith(".fna"))
    if not fna_files:
        print(f"ERROR: No .fna files found in {GENOME_DIR}")
        print("Run 03_download_gtdb_genomes.sh first.")
        sys.exit(1)

    print(f"Found {len(fna_files)} genome files in {GENOME_DIR}")
    print(f"Extracting 1 contig >= {MIN_CONTIG_LEN:,} bp per genome...")

    written = 0
    skipped_no_contig = 0

    with open(OUTPUT_FASTA, "w") as fout:
        for i, fname in enumerate(fna_files):
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(fna_files)} processed "
                      f"(written={written}, skipped={skipped_no_contig})")

            fpath = os.path.join(GENOME_DIR, fname)
            try:
                contigs = parse_fasta(fpath)
            except Exception as e:
                print(f"  WARNING: Failed to parse {fname}: {e}")
                skipped_no_contig += 1
                continue

            valid = [(h, clean_dna(s)) for h, s in contigs if len(s) >= MIN_CONTIG_LEN]
            if not valid:
                skipped_no_contig += 1
                continue

            header, seq = rng.choice(valid)
            # genome name from filename: GCA_000006945.2_genomic.fna -> GCA_000006945.2
            genome_name = fname.replace("_genomic.fna", "")
            fout.write(f">gtdb|{genome_name}|{header}\n")
            for j in range(0, len(seq), 80):
                fout.write(seq[j:j+80] + "\n")
            written += 1

    total_bp = 0
    with open(OUTPUT_FASTA) as f:
        seq = []
        for line in f:
            if line.startswith(">"):
                if seq:
                    total_bp += len("".join(seq))
                seq = []
            else:
                seq.append(line.strip())
        if seq:
            total_bp += len("".join(seq))

    print(f"\nDone.")
    print(f"  Written:  {written}")
    print(f"  Skipped:  {skipped_no_contig} (no contig >= {MIN_CONTIG_LEN:,} bp)")
    print(f"  Total bp: {total_bp:,} ({total_bp/1e6:.1f} Mb)")
    print(f"  Output:   {OUTPUT_FASTA}")


if __name__ == "__main__":
    main()
