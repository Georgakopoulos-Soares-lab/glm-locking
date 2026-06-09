#!/usr/bin/env python3
"""Three-way accession-disjoint genus-stratified split for clean re-evaluation.

Produces train / validation / final-test partitions with NO taxonomic (genus)
overlap across any partition.  The final-test set is touched EXACTLY ONCE.

Outputs (in experiments/split_manifest/):
  train_accessions.txt        — accessions for training
  val_accessions.txt          — accessions for validation / checkpoint selection
  final_test_accessions.txt   — accessions for final reporting (touched once)
  split_report.txt            — genus counts, size statistics
  train.fasta / val.fasta / final_test.fasta

Usage:
  python3 experiments/split_manifest/build_three_way_split.py \
      --input data/attack.fasta \
      --train-frac 0.70 --val-frac 0.15 --test-frac 0.15 \
      --seed 42
"""

import os, sys, argparse, random, re
from collections import defaultdict


GENUS_RE = re.compile(r"\s+(virus|viridae|virinae)$", re.IGNORECASE)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def extract_genus(header: str) -> str:
    """Token 1 of the header is the genus-level grouping token."""
    parts = header.split()
    if len(parts) < 2:
        return parts[0].lower()
    return parts[1].lower()


def extract_accession(header: str) -> str:
    """Token 0 is the NCBI accession."""
    return header.split()[0] if header else "unknown"


def write_fasta(records, path):
    with open(path, "w") as f:
        for hdr, seq in records:
            f.write(f">{hdr}\n{seq}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/attack.fasta")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac",   type=float, default=0.15)
    p.add_argument("--test-frac",  type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) > 0.001:
        raise ValueError("Fractions must sum to 1.0")

    random.seed(args.seed)

    # Load & group by genus
    by_genus = defaultdict(list)
    total = 0
    for hdr, seq in parse_fasta(args.input):
        g = extract_genus(hdr)
        by_genus[g].append((hdr, seq))
        total += 1

    genera = list(by_genus.keys())
    random.shuffle(genera)
    n_gen = len(genera)

    n_train_gen = max(1, int(n_gen * args.train_frac))
    n_val_gen   = max(1, int(n_gen * args.val_frac))
    # test gets the rest
    train_genera = set(genera[:n_train_gen])
    val_genera   = set(genera[n_train_gen:n_train_gen + n_val_gen])
    test_genera  = set(genera[n_train_gen + n_val_gen:])

    train_recs = []; val_recs = []; test_recs = []
    for g in train_genera:
        train_recs.extend(by_genus[g])
    for g in val_genera:
        val_recs.extend(by_genus[g])
    for g in test_genera:
        test_recs.extend(by_genus[g])

    # Verify no genus overlap
    assert train_genera.isdisjoint(val_genera)
    assert train_genera.isdisjoint(test_genera)
    assert val_genera.isdisjoint(test_genera)

    # Write accessions
    for label, recs in [("train", train_recs), ("val", val_recs), ("final_test", test_recs)]:
        accs = sorted(set(extract_accession(h) for h, _ in recs))
        with open(os.path.join(OUT_DIR, f"{label}_accessions.txt"), "w") as f:
            f.write("\n".join(accs) + "\n")
        write_fasta(recs, os.path.join(OUT_DIR, f"{label}.fasta"))

    # Report
    report_path = os.path.join(OUT_DIR, "split_report.txt")
    with open(report_path, "w") as f:
        f.write(f"input={args.input}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"train_frac={args.train_frac} val_frac={args.val_frac} test_frac={args.test_frac}\n")
        f.write(f"total_seqs={total} total_genera={n_gen}\n")
        f.write(f"train_seqs={len(train_recs)} train_genera={len(train_genera)}\n")
        f.write(f"val_seqs={len(val_recs)} val_genera={len(val_genera)}\n")
        f.write(f"final_test_seqs={len(test_recs)} final_test_genera={len(test_genera)}\n")
        f.write(f"\nTrain genera: {sorted(train_genera)}\n")
        f.write(f"Val genera: {sorted(val_genera)}\n")
        f.write(f"Final-test genera: {sorted(test_genera)}\n")

    print(f"Split complete: {len(train_recs)} train / {len(val_recs)} val / {len(test_recs)} final-test")
    print(f"  {len(train_genera)} train genera / {len(val_genera)} val genera / {len(test_genera)} test genera")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
