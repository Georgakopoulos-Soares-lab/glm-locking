#!/usr/bin/env python3
"""Automated download of ~10,000 viral genomes from NCBI for the 10k stress test.

Uses NCBI Entrez API (Biopython). Respects rate limits.
Downloads per-family FASTAs, then merge_10k_corpus.py does the rest.

Usage:
    conda run -n evo python experiments/exp1_datascale/download_10k_viruses.py

Output: experiments/exp1_datascale/downloads/<family>.fasta
"""

import os, sys, time, argparse
from pathlib import Path

# Add repo root to path for Biopython access
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from Bio import Entrez
from Bio import SeqIO

Entrez.email = "arisk@example.com"  # NCBI requires an email
Entrez.tool = "glm-locking-10k"

OUT_DIR = os.path.join(REPO_ROOT, "experiments", "exp1_datascale", "downloads")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Virus families to target (human-pathogenic, excluded from OpenGenome) ──
# Format: (family_name, entrez_search_term, target_count)
# Target ~10,000 total minus the existing 910
VIRUS_FAMILIES = [
    ("Influenza_A", '"Influenza A virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 1500),
    ("SARS-CoV-2", '"Severe acute respiratory syndrome coronavirus 2"[Organism] AND complete genome[All Fields] AND refseq[filter]', 1200),
    ("HIV", '"Human immunodeficiency virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 1000),
    ("Hepatitis_C", '"Hepatitis C virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 800),
    ("Ebola_Marburg", '(Ebola virus[Organism] OR Ebolavirus[Organism] OR Marburg virus[Organism] OR Marburg marburgvirus[Organism]) AND complete genome[All Fields] AND refseq[filter]', 500),
    ("Dengue", '"Dengue virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 600),
    ("Zika", '"Zika virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 400),
    ("West_Nile", '"West Nile virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 400),
    ("Herpesviridae", 'Herpesviridae[Organism] AND complete genome[All Fields] AND refseq[filter]', 600),
    ("Poxviridae", 'Poxviridae[Organism] AND complete genome[All Fields] AND refseq[filter]', 400),
    ("Adenoviridae", 'Adenoviridae[Organism] AND complete genome[All Fields] AND refseq[filter]', 400),
    ("Papillomaviridae", 'Papillomaviridae[Organism] AND complete genome[All Fields] AND refseq[filter]', 400),
    ("Coronaviridae_other", 'Coronaviridae[Organism] NOT "Severe acute respiratory syndrome coronavirus 2"[Organism] AND complete genome[All Fields] AND refseq[filter]', 300),
    ("Paramyxoviridae", 'Paramyxoviridae[Organism] AND complete genome[All Fields] AND refseq[filter]', 300),
    ("Rhabdoviridae", 'Rhabdoviridae[Organism] AND complete genome[All Fields] AND refseq[filter]', 200),
    ("Caliciviridae", 'Caliciviridae[Organism] AND complete genome[All Fields] AND refseq[filter]', 200),
    ("Orthomyxoviridae_other", 'Orthomyxoviridae[Organism] NOT "Influenza A virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 200),
    ("Retroviridae_other", 'Retroviridae[Organism] NOT "Human immunodeficiency virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 200),
    ("Yellow_fever", '"Yellow fever virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 150),
    ("Chikungunya", '"Chikungunya virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 150),
    ("Measles", '"Measles virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 100),
    ("Nipah_Hendra", '(Nipah virus[Organism] OR Hendra virus[Organism]) AND complete genome[All Fields] AND refseq[filter]', 50),
    ("Lassa", '"Lassa virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 50),
    ("Japanese_encephalitis", '"Japanese encephalitis virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 100),
    ("Tick_borne_encephalitis", '"Tick-borne encephalitis virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 100),
    ("Rift_Valley_fever", '"Rift Valley fever virus"[Organism] AND complete genome[All Fields] AND refseq[filter]', 50),
    ("Rotavirus", 'Rotavirus[Organism] AND complete genome[All Fields] AND refseq[filter]', 200),
    ("Enterovirus", 'Enterovirus[Organism] AND complete genome[All Fields] AND refseq[filter]', 300),
    ("Norovirus", 'Norovirus[Organism] AND complete genome[All Fields] AND refseq[filter]', 200),
]


def search_and_fetch(family_name, query, target_count, batch_size=200):
    """Search NCBI and fetch FASTA sequences, respecting rate limits."""
    out_path = os.path.join(OUT_DIR, f"{family_name}.fasta")
    
    if os.path.exists(out_path):
        n_existing = sum(1 for _ in SeqIO.parse(out_path, "fasta"))
        if n_existing >= target_count:
            print(f"  [{family_name}] Already have {n_existing} sequences (target {target_count}), skipping")
            return n_existing
        print(f"  [{family_name}] Existing: {n_existing}, need: {target_count}")
    else:
        print(f"  [{family_name}] Target: {target_count}")

    # Step 1: Search for IDs
    print(f"    Searching NCBI...")
    try:
        handle = Entrez.esearch(
            db="nucleotide",
            term=query,
            retmax=target_count,
            usehistory="y",
            idtype="acc",
        )
        search_results = Entrez.read(handle)
        handle.close()
    except Exception as e:
        print(f"    ERROR during search: {e}")
        return 0

    count = int(search_results["Count"])
    id_list = search_results.get("IdList", [])
    webenv = search_results.get("WebEnv", "")
    query_key = search_results.get("QueryKey", "")
    
    print(f"    Found {count} total, will fetch up to {target_count}")
    
    if count == 0:
        print(f"    No results, skipping")
        return 0

    # Step 2: Fetch in batches
    total_fetched = 0
    retstart = 0
    max_retries = 3
    
    with open(out_path, "w") as f_out:
        while total_fetched < min(count, target_count):
            batch_size_actual = min(batch_size, target_count - total_fetched)
            
            for attempt in range(max_retries):
                try:
                    if webenv and query_key:
                        handle = Entrez.efetch(
                            db="nucleotide",
                            rettype="fasta",
                            retmode="text",
                            retstart=retstart,
                            retmax=batch_size_actual,
                            webenv=webenv,
                            query_key=query_key,
                        )
                    else:
                        handle = Entrez.efetch(
                            db="nucleotide",
                            id=",".join(id_list[retstart:retstart + batch_size_actual]),
                            rettype="fasta",
                            retmode="text",
                        )
                    
                    data = handle.read()
                    handle.close()
                    
                    if data:
                        f_out.write(data)
                        f_out.flush()
                        total_fetched += batch_size_actual
                        retstart += batch_size_actual
                        print(f"    Fetched {total_fetched}/{min(count, target_count)}")
                        break
                    else:
                        raise ValueError("Empty response")
                        
                except Exception as e:
                    print(f"    Attempt {attempt+1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt * 5
                        print(f"    Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"    GIVING UP on this batch")
                        # Write what we have
                        f_out.flush()
                        total_fetched = min(count, target_count)  # break outer loop
            
            # Respect NCBI rate limit: max 3 requests/sec without API key
            time.sleep(0.4)

    n_final = sum(1 for _ in SeqIO.parse(out_path, "fasta"))
    print(f"  [{family_name}] Done: {n_final} sequences saved")
    return n_final


def main():
    parser = argparse.ArgumentParser(description="Download 10k viral genomes from NCBI")
    parser.add_argument("--family", type=str, default=None, 
                        help="Download only a specific family (by name)")
    args = parser.parse_args()

    print("=" * 70)
    print("10k Virus Genome Download — NCBI Entrez API")
    print("=" * 70)
    print(f"Output directory: {OUT_DIR}")
    print(f"Rate limit: ~3 requests/sec (0.4s delay between requests)")
    print(f"Estimated time: ~10-15 minutes for full download")
    print()

    families = VIRUS_FAMILIES
    if args.family:
        families = [(f_name, q, t) for f_name, q, t in VIRUS_FAMILIES if args.family.lower() in f_name.lower()]
        if not families:
            print(f"Family '{args.family}' not found. Available: {[f[0] for f in VIRUS_FAMILIES]}")
            return

    total_downloaded = 0
    for family_name, query, target in families:
        n = search_and_fetch(family_name, query, target)
        total_downloaded += n
        print()

    print("=" * 70)
    print(f"TOTAL downloaded: {total_downloaded} sequences across {len(families)} families")
    print(f"Next step: python experiments/exp1_datascale/merge_10k_corpus.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
