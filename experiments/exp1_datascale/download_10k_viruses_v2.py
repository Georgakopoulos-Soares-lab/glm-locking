#!/usr/bin/env python3
"""Download ~10,000 viral genomes from NCBI using simple organism queries.

Uses urllib directly (avoids Biopython issues). Simple organism/taxonomy
queries. Filters by minimum sequence length to get complete/near-complete genomes.

Usage:
    python3 experiments/exp1_datascale/download_10k_viruses_v2.py

Output: experiments/exp1_datascale/downloads/<family>.fasta
"""

import os, sys, time, urllib.request, urllib.parse, gzip, io
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "exp1_datascale", "downloads")
os.makedirs(OUT_DIR, exist_ok=True)

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ── Virus taxa to download (simple organism/taxon queries) ──
# target = desired number of sequences after length filtering
VIRUS_TAXA = [
    # dsDNA viruses
    ("Herpesviridae", "Herpesviridae[Organism]", 600),
    ("Poxviridae", "Poxviridae[Organism]", 400),
    ("Adenoviridae", "Adenoviridae[Organism]", 400),
    ("Papillomaviridae", "Papillomaviridae[Organism]", 500),
    ("Polyomaviridae", "Polyomaviridae[Organism]", 200),
    ("Mimiviridae", "Mimiviridae[Organism]", 50),
    ("Iridoviridae", "Iridoviridae[Organism]", 100),
    ("Baculoviridae", "Baculoviridae[Organism]", 100),
    ("Asfarviridae", "Asfarviridae[Organism]", 50),
    # ssRNA(+) viruses
    ("Flaviviridae", "Flaviviridae[Organism]", 800),
    ("Coronaviridae", "Coronaviridae[Organism]", 800),
    ("Picornaviridae", "Picornaviridae[Organism]", 400),
    ("Togaviridae", "Togaviridae[Organism]", 200),
    ("Caliciviridae", "Caliciviridae[Organism]", 200),
    ("Astroviridae", "Astroviridae[Organism]", 150),
    # ssRNA(-) viruses
    ("Paramyxoviridae", "Paramyxoviridae[Organism]", 400),
    ("Filoviridae", "Filoviridae[Organism]", 200),
    ("Rhabdoviridae", "Rhabdoviridae[Organism]", 200),
    ("Orthomyxoviridae", "Orthomyxoviridae[Organism]", 600),
    ("Bunyaviridae", "Bunyaviridae[Organism]", 150),
    ("Arenaviridae", "Arenaviridae[Organism]", 150),
    ("Hantaviridae", "Hantaviridae[Organism]", 150),
    ("Peribunyaviridae", "Peribunyaviridae[Organism]", 100),
    ("Phenuiviridae", "Phenuiviridae[Organism]", 100),
    ("Nairoviridae", "Nairoviridae[Organism]", 100),
    # dsRNA viruses
    ("Reoviridae", "Reoviridae[Organism]", 300),
    ("Birnaviridae", "Birnaviridae[Organism]", 100),
    # Retroviruses
    ("Retroviridae", "Retroviridae[Organism]", 600),
    # ssDNA viruses
    ("Parvoviridae", "Parvoviridae[Organism]", 200),
    ("Circoviridae", "Circoviridae[Organism]", 150),
    ("Anelloviridae", "Anelloviridae[Organism]", 200),
    # Hepatitis
    ("Hepadnaviridae", "Hepadnaviridae[Organism]", 200),
    # Other human-pathogenic specific
    ("Ebolavirus", "Ebolavirus[Organism]", 200),
    ("Marburgvirus", "Marburgvirus[Organism]", 50),
    ("Lyssavirus", "Lyssavirus[Organism]", 150),
    ("Enterovirus", "Enterovirus[Organism]", 300),
    ("Norovirus", "Norovirus[Organism]", 200),
    ("Rotavirus", "Rotavirus[Organism]", 200),
    ("Alphavirus", "Alphavirus[Organism]", 150),
    ("Lentivirus", "Lentivirus[Organism]", 200),
]

MIN_SEQ_LENGTH = 500  # minimum bp for a usable genome segment


def ncbi_request(url, timeout=30, max_retries=3):
    """Make an NCBI API request with retries."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "glm-locking/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                # Handle gzip
                if resp.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return data.decode("utf-8")
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 3
                print(f"    Retry {attempt+1}/{max_retries} in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def search_ncbi(query, retmax=10000):
    """Search NCBI nucleotide database, return WebEnv, QueryKey, Count."""
    params = urllib.parse.urlencode({
        "db": "nucleotide",
        "term": query,
        "retmax": retmax,
        "usehistory": "y",
    })
    url = f"{ESEARCH}?{params}"
    xml_str = ncbi_request(url)
    
    # Simple XML parsing (avoid xml.etree for robustness)
    import re
    count = 0
    webenv = ""
    qkey = ""
    
    m = re.search(r"<Count>(\d+)</Count>", xml_str)
    if m:
        count = int(m.group(1))
    m = re.search(r"<WebEnv>(\S+)</WebEnv>", xml_str)
    if m:
        webenv = m.group(1)
    m = re.search(r"<QueryKey>(\d+)</QueryKey>", xml_str)
    if m:
        qkey = m.group(1)
    
    return webenv, qkey, count


def fetch_fasta(webenv, qkey, retstart, retmax):
    """Fetch FASTA records from NCBI."""
    params = urllib.parse.urlencode({
        "db": "nucleotide",
        "query_key": qkey,
        "WebEnv": webenv,
        "rettype": "fasta",
        "retmode": "text",
        "retstart": retstart,
        "retmax": retmax,
    })
    url = f"{EFETCH}?{params}"
    return ncbi_request(url, timeout=120)


def download_family(name, query, target):
    """Download sequences for a virus family/taxon."""
    out_path = os.path.join(OUT_DIR, f"{name}.fasta")
    
    # Check existing
    if os.path.exists(out_path):
        n = count_fasta_seqs(out_path)
        if n >= target:
            print(f"  [{name}] Already have {n} (target {target}), skip")
            return n
    
    print(f"  [{name}] Searching... (target {target})", end=" ", flush=True)
    
    try:
        webenv, qkey, count = search_ncbi(query, retmax=min(target * 10, 50000))
    except Exception as e:
        print(f"SEARCH FAILED: {e}")
        return 0
    
    print(f"Found {count} total", end="", flush=True)
    
    if count == 0:
        print(", skipping")
        return 0
    
    # Need to fetch more than target since we'll filter by length
    # Fetch up to target*5 to have enough after filtering
    fetch_total = min(count, target * 5)
    batch_size = 200
    all_fasta = ""
    fetched = 0
    
    print(f", fetching up to {fetch_total}...", end=" ", flush=True)
    
    while fetched < fetch_total:
        batch = min(batch_size, fetch_total - fetched)
        try:
            fasta_text = fetch_fasta(webenv, qkey, fetched, batch)
            all_fasta += fasta_text
            fetched += batch
            if fetched % 1000 == 0:
                print(f"{fetched}...", end="", flush=True)
        except Exception as e:
            print(f"FETCH ERROR at {fetched}: {e}")
            time.sleep(2)
            continue
        time.sleep(0.3)  # Rate limit
    
    print(f" done ({fetched} fetched)", end="", flush=True)
    
    # Filter by length and write
    records = parse_fasta_text(all_fasta)
    filtered = [(h, s) for h, s in records if len(s) >= MIN_SEQ_LENGTH]
    
    # Take up to target
    filtered = filtered[:target]
    
    with open(out_path, "w") as f:
        for header, seq in filtered:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")
    
    print(f" -> saved {len(filtered)} sequences (>= {MIN_SEQ_LENGTH}bp)")
    return len(filtered)


def parse_fasta_text(text):
    """Parse FASTA text into list of (header, seq) tuples."""
    records = []
    header = None
    seq_parts = []
    for line in text.split("\n"):
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


def count_fasta_seqs(path):
    """Count sequences in a FASTA file."""
    n = 0
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(">"):
                    n += 1
    except:
        pass
    return n


def main():
    print("=" * 70)
    print("10k Virus Genome Download v2 — NCBI eutils")
    print(f"Output: {OUT_DIR}")
    print(f"Taxa to query: {len(VIRUS_TAXA)}")
    print("=" * 70)
    
    total = 0
    for name, query, target in VIRUS_TAXA:
        n = download_family(name, query, target)
        total += n
    
    print()
    print("=" * 70)
    print(f"TOTAL: {total} sequences downloaded")
    print(f"Next: python experiments/exp1_datascale/merge_10k_corpus.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
