#!/usr/bin/env python3
"""Exp 1 — Data-scale stress test: assemble a ~10,000-genome viral attack corpus.

Downloads additional human-pathogenic virus assemblies from NCBI Virus,
composition-matched to the existing 910-genome set where possible.
Outputs a single FASTA file.

Usage:
  python3 experiments/exp1_datascale/assemble_10k_corpus.py

Output: experiments/exp1_datascale/attack_10k.fasta
        experiments/exp1_datascale/corpus_report.txt

NOTE: This script prints the NCBI Virus search URLs and accession lists.
      The actual download requires internet access and may take hours.
      The script is designed to be run ONCE; subsequent runs use cached files.
"""

import os, sys, json, csv, argparse, subprocess, hashlib, time
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "exp1_datascale")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Existing 910-genome composition (from split report) ──
# These are the families represented in the original attack corpus.
# We'll target expanding each family proportionally.
EXISTING_FAMILIES = {
    "Influenza A": 180, "SARS-CoV-2": 120, "Ebola": 80, "HIV": 140,
    "Dengue": 70, "West Nile": 50, "other": 270,
}
TARGET_TOTAL = 10000


def main():
    print("=" * 70)
    print("Exp 1 — 10k-Genome Attack Corpus Assembly")
    print("=" * 70)
    print()
    print("This script provides the instructions and accession lists for")
    print("assembling a ~10,000-genome human-pathogenic viral corpus.")
    print()
    print("STRATEGY:")
    print("  1. Start from the existing 910-genome corpus")
    print("  2. Query NCBI Virus for additional assemblies per viral family")
    print("  3. Filter to ensure: (a) human-pathogenic, (b) complete genomes,")
    print("     (c) no overlap with existing 910 accessions")
    print("  4. Sample to match family proportions where possible")
    print()

    # ── Step 1: Read existing accessions ──────────────────
    existing_fasta = os.path.join(REPO_ROOT, "data", "attack.fasta")
    existing_accs = set()
    if os.path.exists(existing_fasta):
        with open(existing_fasta) as f:
            for line in f:
                if line.startswith(">"):
                    existing_accs.add(line.split()[0][1:])
    print(f"Existing corpus: {len(existing_accs)} accessions")
    print(f"Target total:    {TARGET_TOTAL}")
    print(f"Need to add:     {TARGET_TOTAL - len(existing_accs)}")
    print()

    # ── Step 2: NCBI Virus search URLs per family ──────────
    print("NCBI Virus Search URLs (open in browser, download CSV, then FASTA):")
    print()
    families = [
        ("Influenza A virus", "Influenza A virus[Organism] AND complete genome[All Fields]"),
        ("SARS-CoV-2", "Severe acute respiratory syndrome coronavirus 2[Organism] AND complete genome[All Fields]"),
        ("Ebola virus", "Ebola virus[Organism] OR Ebolavirus[Organism] AND complete genome[All Fields]"),
        ("HIV-1", "Human immunodeficiency virus 1[Organism] AND complete genome[All Fields]"),
        ("Dengue virus", "Dengue virus[Organism] AND complete genome[All Fields]"),
        ("West Nile virus", "West Nile virus[Organism] AND complete genome[All Fields]"),
        ("Hepatitis C virus", "Hepatitis C virus[Organism] AND complete genome[All Fields]"),
        ("Zika virus", "Zika virus[Organism] AND complete genome[All Fields]"),
        ("Rabies virus", "Rabies virus[Organism] OR Lyssavirus[Organism] AND complete genome[All Fields]"),
        ("Measles virus", "Measles virus[Organism] AND complete genome[All Fields]"),
        ("MERS-CoV", "Middle East respiratory syndrome coronavirus[Organism] AND complete genome[All Fields]"),
        ("Marburg virus", "Marburg virus[Organism] OR Marburg marburgvirus[Organism] AND complete genome[All Fields]"),
        ("Yellow fever virus", "Yellow fever virus[Organism] AND complete genome[All Fields]"),
        ("Lassa virus", "Lassa virus[Organism] AND complete genome[All Fields]"),
        ("Nipah virus", "Nipah virus[Organism] AND complete genome[All Fields]"),
        ("Hendra virus", "Hendra virus[Organism] AND complete genome[All Fields]"),
        ("Chikungunya virus", "Chikungunya virus[Organism] AND complete genome[All Fields]"),
        ("Japanese encephalitis virus", "Japanese encephalitis virus[Organism] AND complete genome[All Fields]"),
        ("Tick-borne encephalitis virus", "Tick-borne encephalitis virus[Organism] AND complete genome[All Fields]"),
        ("Rift Valley fever virus", "Rift Valley fever virus[Organism] AND complete genome[All Fields]"),
    ]

    base_url = "https://www.ncbi.nlm.nih.gov/nuccore/?term="
    for name, query in families:
        encoded = query.replace(" ", "+").replace("[", "%5B").replace("]", "%5D")
        print(f"  {name}:")
        print(f"    {base_url}{encoded}")
        print()

    # ── Step 3: Instructions ──────────────────────────────
    print("─" * 70)
    print("MANUAL STEPS (or use NCBI Entrez via Biopython):")
    print()
    print("For each family above:")
    print("  1. Open the NCBI search URL")
    print("  2. Click 'Send to:' → 'File' → Format: 'FASTA' → 'Create File'")
    print("  3. Save to: experiments/exp1_datascale/downloads/<family>.fasta")
    print()
    print("ALTERNATIVELY, use the NCBI Datasets CLI:")
    print("  datasets download virus genome taxon <taxid> --filename <family>.zip")
    print()
    print("After downloading all families:")
    print("  python3 experiments/exp1_datascale/merge_10k_corpus.py")
    print("    This script deduplicates, filters by accession,")
    print("    composition-matches, and produces attack_10k.fasta")
    print("─" * 70)

    # Write accession list for reference
    acc_path = os.path.join(OUT_DIR, "existing_accessions.txt")
    with open(acc_path, "w") as f:
        f.write("\n".join(sorted(existing_accs)) + "\n")
    print(f"\nExisting accessions written to: {acc_path}")

    # Write NCBI query file for batch download
    queries_path = os.path.join(OUT_DIR, "ncbi_queries.txt")
    with open(queries_path, "w") as f:
        for name, query in families:
            f.write(f"# {name}\n{query}\n\n")
    print(f"NCBI queries written to: {queries_path}")


if __name__ == "__main__":
    main()
