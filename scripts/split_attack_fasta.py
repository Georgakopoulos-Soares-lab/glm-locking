"""Genus-stratified split of attack.fasta into train + held-out viral sets.

Header format is: '>NCxxxxxx Genus species ...'
We split at the genus level so the held-out set tests true generalization, not
just memorization of within-genus patterns.

Outputs:
    data/attack_train.fasta    — 70% of genera (random per --seed)
    data/attack_heldout.fasta  — 30% of genera (unseen during FT)
"""
import os, sys, argparse, random, re
from collections import defaultdict

GENUS_SUFFIX_RE = re.compile(r"\s+(virus|viridae|virinae)$", re.IGNORECASE)


def parse_fasta(path):
    """Yield (header_full, seq_str) pairs."""
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
    """Take first 1-2 tokens of the description as taxonomic key."""
    parts = header.split()
    if len(parts) < 2: return parts[0]
    # token 0 = NC accession; token 1 = genus token (e.g. "Macropodid")
    # token 2 may be species/host; group by token 1 only for genus-level holdout
    return parts[1].lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/attack.fasta")
    p.add_argument("--train-out",  default="data/attack_train.fasta")
    p.add_argument("--heldout-out", default="data/attack_heldout.fasta")
    p.add_argument("--heldout-frac", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", default="data/attack_split_report.txt")
    args = p.parse_args()

    random.seed(args.seed)
    by_genus = defaultdict(list)
    total = 0
    for hdr, seq in parse_fasta(args.input):
        g = extract_genus(hdr)
        by_genus[g].append((hdr, seq))
        total += 1
    print(f"Loaded {total} sequences across {len(by_genus)} genus groups")

    genera = sorted(by_genus)
    random.shuffle(genera)
    n_holdout = max(1, int(round(len(genera) * args.heldout_frac)))
    holdout_genera = set(genera[:n_holdout])
    train_genera   = set(genera[n_holdout:])

    with open(args.train_out, "w") as f_tr, open(args.heldout_out, "w") as f_ho:
        n_tr = n_ho = 0
        for g in genera:
            target = f_ho if g in holdout_genera else f_tr
            for hdr, seq in by_genus[g]:
                target.write(">" + hdr + "\n")
                for i in range(0, len(seq), 80):
                    target.write(seq[i:i+80] + "\n")
                if g in holdout_genera: n_ho += 1
                else:                   n_tr += 1

    print(f"Train:   {n_tr} seqs / {len(train_genera)} genera -> {args.train_out}")
    print(f"Holdout: {n_ho} seqs / {len(holdout_genera)} genera -> {args.heldout_out}")

    with open(args.report, "w") as f:
        f.write(f"input={args.input}\nseed={args.seed}\nholdout_frac={args.heldout_frac}\n")
        f.write(f"total_seqs={total} total_genera={len(genera)}\n")
        f.write(f"train_seqs={n_tr} train_genera={len(train_genera)}\n")
        f.write(f"heldout_seqs={n_ho} heldout_genera={len(holdout_genera)}\n\n")
        f.write("Holdout genera:\n  " + "\n  ".join(sorted(holdout_genera)) + "\n\n")
        f.write("Train genera:\n  "   + "\n  ".join(sorted(train_genera))   + "\n")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
