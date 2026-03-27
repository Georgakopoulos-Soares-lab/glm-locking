#!/usr/bin/env python3
"""
Step 2: Sample GTDB genomes for the retain dataset.

Reads GTDB r220 taxonomy files, then samples:
  - ~5,000 bacteria (proportional across phyla)
  - ~500 archaea (proportional across phyla)

For each selected genome, extracts 1 random contig >= 10 kb from the
compressed FASTA and writes it to the output.

Usage:
    python data/download_scripts/02_sample_gtdb.py

Environment variables:
    SCRATCH_DIR  - root scratch directory (default: /scratch/10906/arisk/evo_locking_data)
"""

import gzip
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ===========================================================================
# Config
# ===========================================================================
SEED = 42
N_BACTERIA = 5000
N_ARCHAEA = 500
MIN_CONTIG_LEN = 10_000  # 10 kb

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch/10906/arisk/evo_locking_data")
GTDB_DIR = os.path.join(SCRATCH_DIR, "gtdb")

BAC_TAXONOMY = os.path.join(GTDB_DIR, "bac120_taxonomy_r220.tsv")
ARC_TAXONOMY = os.path.join(GTDB_DIR, "ar53_taxonomy_r220.tsv")
GENOME_BASE = os.path.join(GTDB_DIR, "gtdb_genomes_reps_r220")

OUTPUT_DIR = os.path.join(SCRATCH_DIR, "retain_build")
OUTPUT_FASTA = os.path.join(OUTPUT_DIR, "gtdb_sampled.fasta")
OUTPUT_TSV = os.path.join(OUTPUT_DIR, "gtdb_sampled_metadata.tsv")


# ===========================================================================
# Helpers
# ===========================================================================
def parse_taxonomy(path: str) -> dict[str, dict]:
    """Parse GTDB taxonomy TSV -> {accession: {domain, phylum, ...}}"""
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
            rec = {}
            for rank, level in zip(ranks, levels):
                rec[rank] = level.split("__", 1)[1] if "__" in level else level
            rec["gtdb_accession"] = acc
            # RS_GCF_000006945.2 -> GCF_000006945.2
            ncbi_acc = acc.split("_", 1)[1] if acc.startswith(("RS_", "GB_")) else acc
            rec["ncbi_accession"] = ncbi_acc
            records[acc] = rec
    return records


def find_genome_fasta(ncbi_acc: str, genome_base: str) -> str | None:
    """Locate the .fna.gz file for a genome accession in the GTDB directory tree.

    GTDB r220 stores genomes as:
      database/GCA/000/008/085/GCA_000008085.1_genomic.fna.gz
    or
      database/GCF/000/006/945/GCF_000006945.2_genomic.fna.gz
    """
    # Parse accession: GCA_000008085.1 -> (GCA, 000, 008, 085)
    prefix, number_version = ncbi_acc.split("_", 1)
    number = number_version.split(".")[0]  # e.g. "000008085"

    # Pad to 9 digits
    number = number.zfill(9)
    d1, d2, d3 = number[:3], number[3:6], number[6:9]

    fna_name = f"{ncbi_acc}_genomic.fna.gz"
    path = os.path.join(genome_base, "database", prefix, d1, d2, d3, fna_name)

    if os.path.exists(path):
        return path
    return None


def parse_gzipped_fasta(path: str) -> list[tuple[str, str]]:
    """Parse a gzipped FASTA -> [(header, sequence), ...]"""
    records = []
    header = None
    seq_parts = []
    with gzip.open(path, "rt") as f:
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


def sample_across_phyla(records: dict, n_target: int, rng: random.Random) -> list[dict]:
    """Sample n_target genomes proportionally across phyla."""
    by_phylum: dict[str, list[dict]] = defaultdict(list)
    for rec in records.values():
        p = rec.get("phylum", "")
        if p:
            by_phylum[p].append(rec)

    total = sum(len(v) for v in by_phylum.values())
    if total == 0:
        return []

    # Proportional allocation (at least 1 per phylum)
    allocations = {}
    for phylum, recs in by_phylum.items():
        allocations[phylum] = max(1, int((len(recs) / total) * n_target))

    # Adjust to hit target
    current = sum(allocations.values())
    sorted_phyla = sorted(by_phylum.keys(), key=lambda p: len(by_phylum[p]), reverse=True)
    if current < n_target:
        for p in sorted_phyla:
            diff = min(n_target - current, len(by_phylum[p]) - allocations[p])
            if diff > 0:
                allocations[p] += diff
                current += diff
            if current >= n_target:
                break
    elif current > n_target:
        for p in reversed(sorted_phyla):
            diff = min(current - n_target, allocations[p] - 1)
            if diff > 0:
                allocations[p] -= diff
                current -= diff
            if current <= n_target:
                break

    sampled = []
    for phylum, recs in by_phylum.items():
        n = min(allocations.get(phylum, 0), len(recs))
        sampled.extend(rng.sample(recs, n))
    return sampled


# ===========================================================================
# Main
# ===========================================================================
def main():
    rng = random.Random(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Parse taxonomy ---
    for path in [BAC_TAXONOMY, ARC_TAXONOMY]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run 01_download_gtdb.sh first.")
            sys.exit(1)

    print("Parsing bacterial taxonomy...")
    bac = parse_taxonomy(BAC_TAXONOMY)
    print(f"  {len(bac):,} species reps")

    print("Parsing archaeal taxonomy...")
    arc = parse_taxonomy(ARC_TAXONOMY)
    print(f"  {len(arc):,} species reps")

    # --- Sample ---
    print(f"\nSampling {N_BACTERIA} bacteria across phyla...")
    bac_sampled = sample_across_phyla(bac, N_BACTERIA, rng)
    print(f"Sampling {N_ARCHAEA} archaea across phyla...")
    arc_sampled = sample_across_phyla(arc, N_ARCHAEA, random.Random(SEED + 1))

    all_sampled = bac_sampled + arc_sampled
    print(f"Total sampled: {len(all_sampled)}")

    # Show phylum distribution
    bac_phyla = Counter(r["phylum"] for r in bac_sampled)
    print("\nBacterial phyla (top 10):")
    for p, c in bac_phyla.most_common(10):
        print(f"  {p:40s} {c:5d}")

    # --- Extract 1 contig per genome ---
    print(f"\nExtracting 1 contig >= {MIN_CONTIG_LEN:,} bp per genome...")
    print(f"  Genome directory: {GENOME_BASE}")

    written = 0
    skipped_no_file = 0
    skipped_no_contig = 0
    metadata_rows = []

    with open(OUTPUT_FASTA, "w") as fout:
        for i, rec in enumerate(all_sampled):
            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(all_sampled)} "
                      f"(written={written}, skipped_file={skipped_no_file}, "
                      f"skipped_contig={skipped_no_contig})")

            fpath = find_genome_fasta(rec["ncbi_accession"], GENOME_BASE)
            if fpath is None:
                skipped_no_file += 1
                continue

            try:
                contigs = parse_gzipped_fasta(fpath)
            except Exception as e:
                print(f"  WARNING: Failed to parse {fpath}: {e}")
                skipped_no_file += 1
                continue

            valid = [(h, clean_dna(s)) for h, s in contigs if len(s) >= MIN_CONTIG_LEN]
            if not valid:
                skipped_no_contig += 1
                continue

            header, seq = rng.choice(valid)
            fasta_header = f"gtdb|{rec['ncbi_accession']}|{rec['phylum']}|{header}"
            fout.write(f">{fasta_header}\n")
            for j in range(0, len(seq), 80):
                fout.write(seq[j:j+80] + "\n")

            written += 1
            metadata_rows.append(rec)

    print(f"\nGTDB sampling complete:")
    print(f"  Written:          {written}")
    print(f"  Skipped (no file):   {skipped_no_file}")
    print(f"  Skipped (no contig): {skipped_no_contig}")
    print(f"  Output: {OUTPUT_FASTA}")
    print(f"  Size: {os.path.getsize(OUTPUT_FASTA) / 1e6:.1f} MB")

    # --- Write metadata TSV ---
    with open(OUTPUT_TSV, "w") as f:
        f.write("ncbi_accession\tgtdb_accession\tdomain\tphylum\tclass\torder\tfamily\tgenus\tspecies\n")
        for rec in metadata_rows:
            f.write("\t".join([
                rec["ncbi_accession"], rec["gtdb_accession"],
                rec["domain"], rec["phylum"], rec["class"],
                rec["order"], rec["family"], rec["genus"], rec["species"],
            ]) + "\n")
    print(f"  Metadata: {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
