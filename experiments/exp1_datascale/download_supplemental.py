#!/usr/bin/env python3
"""Supplemental download: add ~3,000 more viral genomes to reach 10k total.

Targets families that were under-sampled in the first pass plus new families.
Uses simple organism-name queries that work reliably with NCBI.
"""

import os, sys, time, subprocess, tempfile, shutil

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
os.makedirs(OUT_DIR, exist_ok=True)

# Simple organism queries that work reliably.  Format: (family_name, organism, target)
# Target = how many MORE we want beyond what's already downloaded
FAMILIES = [
    ("Herpesviridae_2", "Herpesviridae[Organism]", 400),
    ("Poxviridae_2", "Poxviridae[Organism]", 200),
    ("Flaviviridae_2", "Flaviviridae[Organism]", 200),
    ("Papillomaviridae_2", "Papillomaviridae[Organism]", 200),
    ("Adenoviridae_2", "Adenoviridae[Organism]", 100),
    ("Coronaviridae_2", "Coronaviridae[Organism]", 100),
    ("Retroviridae", "Retroviridae[Organism]", 200),
    ("Picornaviridae", "Picornaviridae[Organism]", 200),
    ("Togaviridae", "Togaviridae[Organism]", 150),
    ("Astroviridae", "Astroviridae[Organism]", 100),
    ("Arenaviridae", "Arenaviridae[Organism]", 150),
    ("Hepeviridae", "Hepeviridae[Organism]", 100),
    ("Birnaviridae", "Birnaviridae[Organism]", 100),
    ("Asfarviridae", "Asfarviridae[Organism]", 80),
    ("Reoviridae", "Reoviridae[Organism]", 200),
    ("Filoviridae", "Filoviridae[Organism]", 100),
    ("Paramyxoviridae_2", "Paramyxoviridae[Organism]", 150),
    ("Rhabdoviridae_2", "Rhabdoviridae[Organism]", 150),
    ("Bunyaviridae_2", "Bunyaviridae[Organism]", 100),
]


def fetch_via_curl(organism_query, target, out_path):
    """Use curl to NCBI esearch + efetch. Handles binary FASTA correctly."""
    import urllib.parse
    
    # Step 1: esearch
    encoded = urllib.parse.quote(organism_query)
    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=nucleotide&term={encoded}&retmax={target}&usehistory=y"
    )
    
    result = subprocess.run(
        ["curl", "-fsSL", "--max-time", "60", search_url],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"    esearch failed: {result.stderr[:200]}")
        return 0
    
    import re
    webenv_m = re.search(r'<WebEnv>([^<]+)</WebEnv>', result.stdout)
    qkey_m = re.search(r'<QueryKey>([^<]+)</QueryKey>', result.stdout)
    count_m = re.search(r'<Count>([^<]+)</Count>', result.stdout)
    
    if not webenv_m or not qkey_m:
        print(f"    No results found")
        return 0
    
    count = int(count_m.group(1))
    webenv = webenv_m.group(1)
    qkey = qkey_m.group(1)
    
    print(f"    Found {count}, fetching up to {target}")
    
    # Step 2: efetch in chunks of 200
    total = 0
    retstart = 0
    fetch_n = min(target, count)
    
    with open(out_path, "ab") as f:
        while total < fetch_n:
            chunk = min(200, fetch_n - total)
            fetch_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=nucleotide&rettype=fasta&retmode=text"
                f"&retstart={retstart}&retmax={chunk}"
                f"&webenv={webenv}&query_key={qkey}"
            )
            
            for attempt in range(3):
                result = subprocess.run(
                    ["curl", "-fsSL", "--max-time", "120", fetch_url],
                    capture_output=True
                )
                if result.returncode == 0 and result.stdout:
                    f.write(result.stdout)
                    f.flush()
                    total += chunk
                    retstart += chunk
                    break
                else:
                    wait = 3 * (attempt + 1)
                    print(f"    Retry {attempt+1}/3 in {wait}s...")
                    time.sleep(wait)
            else:
                print(f"    FAILED at {total}/{fetch_n}")
                break
            
            time.sleep(0.35)  # Rate limit
    
    return total


def main():
    print("=" * 60)
    print("Supplemental virus download (~3,000 more)")
    print("=" * 60)
    
    total_new = 0
    
    for family, query, target in FAMILIES:
        out_path = os.path.join(OUT_DIR, f"{family}.fasta")
        existing = 0
        if os.path.exists(out_path):
            existing = sum(1 for l in open(out_path) if l.startswith(">"))
        if existing >= target:
            print(f"  [{family}] Already have {existing}, skipping (target {target})")
            continue
        
        needed = target - existing
        print(f"  [{family}] Have {existing}, need {needed} more...")
        
        n = fetch_via_curl(query, needed, out_path)
        print(f"    Downloaded {n}")
        total_new += n
        print()
    
    print("=" * 60)
    print(f"Total new sequences: {total_new}")
    print(f"Next: python experiments/exp1_datascale/merge_10k_corpus.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
