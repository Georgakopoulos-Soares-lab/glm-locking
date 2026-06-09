#!/usr/bin/env python3
"""Exp 1 helper — Merge downloaded family FASTAs into a single 10k corpus.

Usage:
  python3 experiments/exp1_datascale/merge_10k_corpus.py

Input:  experiments/exp1_datascale/downloads/*.fasta  (per-family downloads)
Output: experiments/exp1_datascale/attack_10k.fasta
        experiments/exp1_datascale/corpus_report.txt
"""

import os, sys, random, hashlib
from collections import defaultdict, Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "exp1_datascale")
DOWNLOAD_DIR = os.path.join(OUT_DIR, "downloads")
TARGET_TOTAL = 10000


def parse_fasta(path):
    header = None; chunks = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line: continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]; chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def extract_accession(header: str) -> str:
    return header.split()[0] if header else "unknown"


def extract_genus(header: str) -> str:
    parts = header.split()
    if len(parts) < 2:
        return parts[0].lower()
    return parts[1].lower()


def main():
    # Load existing accessions to avoid overlap
    existing_fasta = os.path.join(REPO_ROOT, "data", "attack.fasta")
    existing_accs = set()
    if os.path.exists(existing_fasta):
        for hdr, _ in parse_fasta(existing_fasta):
            existing_accs.add(extract_accession(hdr))

    print(f"Existing corpus: {len(existing_accs)} accessions")
    print(f"Target total:    {TARGET_TOTAL}")

    # Load downloaded family FASTAs
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"Download directory not found: {DOWNLOAD_DIR}")
        print("Run assemble_10k_corpus.py first, download per-family FASTAs,")
        print(f"and place them in {DOWNLOAD_DIR}")
        return

    fasta_files = sorted([f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(('.fasta', '.fa', '.fna'))])
    if not fasta_files:
        print(f"No FASTA files found in {DOWNLOAD_DIR}")
        return

    print(f"Found {len(fasta_files)} family FASTA files")

    # Collect new accessions, deduplicate, filter overlap
    new_records = []
    seen_accs = set()
    by_genus = defaultdict(list)

    for fname in fasta_files:
        fpath = os.path.join(DOWNLOAD_DIR, fname)
        n_added = 0
        for hdr, seq in parse_fasta(fpath):
            acc = extract_accession(hdr)
            if acc in existing_accs or acc in seen_accs:
                continue
            # Filter: require minimum length (complete genomes are >1kb typically)
            if len(seq) < 500:
                continue
            seen_accs.add(acc)
            genus = extract_genus(hdr)
            by_genus[genus].append((hdr, seq))
            n_added += 1
        print(f"  {fname}: {n_added} new accessions")

    # Build corpus: first take all existing, then add new up to target
    all_existing = []
    if os.path.exists(existing_fasta):
        for hdr, seq in parse_fasta(existing_fasta):
            all_existing.append((hdr, seq))

    # Sample from new records to reach target, preserving genus proportions where possible
    needed = TARGET_TOTAL - len(all_existing)
    if needed <= 0:
        print(f"Already have {len(all_existing)} sequences — no expansion needed")
        return

    # Sort genera by size (largest first) and sample proportionally
    all_new = []
    for genus, recs in sorted(by_genus.items(), key=lambda x: -len(x[1])):
        all_new.extend(recs)

    # Take up to 'needed' new records
    random.seed(42)
    if len(all_new) > needed:
        sampled = random.sample(all_new, needed)
    else:
        sampled = all_new
        print(f"WARNING: Only {len(all_new)} new records available, need {needed}")

    corpus = all_existing + sampled
    random.shuffle(corpus)

    # Write output
    out_path = os.path.join(OUT_DIR, "attack_10k.fasta")
    with open(out_path, "w") as f:
        for hdr, seq in corpus:
            f.write(f">{hdr}\n{seq}\n")

    # Report
    report_path = os.path.join(OUT_DIR, "corpus_report.txt")
    genus_counts = Counter()
    for hdr, _ in corpus:
        genus_counts[extract_genus(hdr)] += 1

    with open(report_path, "w") as f:
        f.write(f"Total sequences: {len(corpus)}\n")
        f.write(f"Existing (from attack.fasta): {len(all_existing)}\n")
        f.write(f"Newly added: {len(sampled)}\n")
        f.write(f"Unique genera: {len(genus_counts)}\n\n")
        f.write("Genus distribution:\n")
        for genus, count in genus_counts.most_common(50):
            f.write(f"  {genus}: {count}\n")

    print(f"\nCorpus written: {out_path} ({len(corpus)} sequences)")
    print(f"Report written: {report_path}")

    # Composition-matching check
    print("\nComposition note: the existing 910-genome set has specific viral-family")
    print("proportions. The 10k corpus may differ. Document any differences in the")
    print("corpus_report.txt and flag as a confound in FINDINGS.md if composition")
    print("differs substantially from the 544-train-genome set.")


if __name__ == "__main__":
    main()
