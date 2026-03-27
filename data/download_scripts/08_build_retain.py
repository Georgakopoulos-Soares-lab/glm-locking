#!/usr/bin/env python3
"""
Step 8: Combine GTDB + IMG/VR samples into the final retain.fasta.

Reads the intermediate FASTA files produced by steps 4 and 7,
shuffles them, and writes the final retain.fasta for locking.

Usage:
    python data/download_scripts/08_build_retain.py

Environment variables:
    SCRATCH_DIR  - root scratch directory (default: /scratch/10906/arisk/evo_locking_data)
"""

import os
import random
import sys
from collections import Counter

SEED = 42
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch/10906/arisk/evo_locking_data")
BUILD_DIR = os.path.join(SCRATCH_DIR, "retain_build")

GTDB_FASTA = os.path.join(BUILD_DIR, "gtdb_sampled.fasta")
IMGVR_FASTA = os.path.join(BUILD_DIR, "imgvr_sampled.fasta")
OUTPUT = os.path.join(SCRATCH_DIR, "retain.fasta")

MIN_USABLE_LEN = 512  # minimum to be useful for locking (seq_len in config)


def parse_fasta(path: str) -> list[tuple[str, str]]:
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
                header = line[1:]
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
    all_records = []

    # --- GTDB ---
    if os.path.exists(GTDB_FASTA):
        print(f"Reading GTDB: {GTDB_FASTA}")
        gtdb = parse_fasta(GTDB_FASTA)
        for header, seq in gtdb:
            seq = clean_dna(seq)
            if len(seq) >= MIN_USABLE_LEN:
                all_records.append((header, seq))
        print(f"  {len(gtdb)} parsed, {sum(1 for h, _ in all_records if h.startswith('gtdb|'))} kept (>= {MIN_USABLE_LEN} bp)")
    else:
        print(f"WARNING: {GTDB_FASTA} not found. Run 02_sample_gtdb.py first.")

    # --- IMG/VR ---
    pre_imgvr = len(all_records)
    if os.path.exists(IMGVR_FASTA):
        print(f"Reading IMG/VR: {IMGVR_FASTA}")
        imgvr = parse_fasta(IMGVR_FASTA)
        for header, seq in imgvr:
            seq = clean_dna(seq)
            if len(seq) >= MIN_USABLE_LEN:
                # Prefix with imgvr| if not already
                if not header.startswith("imgvr|"):
                    header = f"imgvr|{header}"
                all_records.append((header, seq))
        imgvr_kept = len(all_records) - pre_imgvr
        print(f"  {len(imgvr)} parsed, {imgvr_kept} kept (>= {MIN_USABLE_LEN} bp)")
    else:
        print(f"WARNING: {IMGVR_FASTA} not found. Run 07_stream_imgvr_sequences.sh first.")
        print("         Proceeding with GTDB-only retain set.")

    if not all_records:
        print("ERROR: No sequences found. Nothing to write.")
        sys.exit(1)

    # --- Shuffle and write ---
    rng.shuffle(all_records)

    print(f"\nWriting {len(all_records)} sequences to {OUTPUT}...")
    with open(OUTPUT, "w") as f:
        for header, seq in all_records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

    # --- Stats ---
    total_bp = sum(len(s) for _, s in all_records)
    lengths = sorted(len(s) for _, s in all_records)

    sources = Counter()
    for h, _ in all_records:
        if h.startswith("gtdb|"):
            sources["GTDB"] += 1
        elif h.startswith("imgvr|"):
            sources["IMG/VR"] += 1
        else:
            sources["other"] += 1

    print(f"\n{'='*50}")
    print(f"retain.fasta summary")
    print(f"{'='*50}")
    print(f"  Sequences:  {len(all_records):,}")
    print(f"  Total bp:   {total_bp:,} ({total_bp / 1e6:.1f} Mb)")
    print(f"  Min length: {lengths[0]:,} bp")
    print(f"  Max length: {lengths[-1]:,} bp")
    print(f"  Median:     {lengths[len(lengths)//2]:,} bp")
    print(f"  File size:  {os.path.getsize(OUTPUT) / 1e6:.1f} MB")
    print(f"\n  Composition:")
    for source, count in sources.most_common():
        print(f"    {source:10s} {count:,}")

    # --- Symlink into evo-locking/data/ ---
    project_data = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    # Resolve to absolute
    project_data = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    link_path = os.path.join(project_data, "retain.fasta")

    os.makedirs(project_data, exist_ok=True)
    if os.path.exists(link_path) or os.path.islink(link_path):
        os.remove(link_path)
    os.symlink(OUTPUT, link_path)
    print(f"\n  Symlinked: {link_path} -> {OUTPUT}")
    print(f"\nDone. Ready for locking: python scripts/lock.py")


if __name__ == "__main__":
    main()
