#!/usr/bin/env python3
"""Ultra-resilient viral genome downloader for unreliable NCBI API.

Uses aggressive retries, exponential backoff, and fallback strategies.
Downloads ~10k viral genomes for the 10k stress test.

Usage:
    conda run -n evo python experiments/exp1_datascale/download_10k_resilient.py
"""

import os, sys, time, re, gzip, io
import urllib.request
import urllib.parse
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "exp1_datascale", "downloads")
os.makedirs(OUT_DIR, exist_ok=True)

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Taxa: (filename_prefix, simple_organism_name, target_sequences)
# Use SIMPLE organism names (no [Organism] tags — they cause 500s)
TAXA = [
    ("Herpesviridae", "Herpesviridae", 600),
    ("Poxviridae", "Poxviridae", 400),
    ("Adenoviridae", "Adenoviridae", 400),
    ("Papillomaviridae", "Papillomaviridae", 500),
    ("Polyomaviridae", "Polyomaviridae", 200),
    ("Mimiviridae", "Mimiviridae", 50),
    ("Iridoviridae", "Iridoviridae", 100),
    ("Baculoviridae", "Baculoviridae", 100),
    ("Asfarviridae", "Asfarviridae", 50),
    ("Flaviviridae", "Flaviviridae", 800),
    ("Coronaviridae", "Coronaviridae", 800),
    ("Picornaviridae", "Picornaviridae", 400),
    ("Togaviridae", "Togaviridae", 200),
    ("Caliciviridae", "Caliciviridae", 200),
    ("Astroviridae", "Astroviridae", 150),
    ("Paramyxoviridae", "Paramyxoviridae", 400),
    ("Filoviridae", "Filoviridae", 200),
    ("Rhabdoviridae", "Rhabdoviridae", 200),
    ("Orthomyxoviridae", "Orthomyxoviridae", 600),
    ("Bunyaviridae", "Bunyaviridae", 150),
    ("Arenaviridae", "Arenaviridae", 150),
    ("Hantaviridae", "Hantaviridae", 150),
    ("Peribunyaviridae", "Peribunyaviridae", 100),
    ("Phenuiviridae", "Phenuiviridae", 100),
    ("Nairoviridae", "Nairoviridae", 100),
    ("Reoviridae", "Reoviridae", 300),
    ("Birnaviridae", "Birnaviridae", 100),
    ("Retroviridae", "Retroviridae", 600),
    ("Parvoviridae", "Parvoviridae", 200),
    ("Circoviridae", "Circoviridae", 150),
    ("Anelloviridae", "Anelloviridae", 200),
    ("Hepadnaviridae", "Hepadnaviridae", 200),
    ("Ebolavirus", "Ebolavirus", 200),
    ("Marburgvirus", "Marburgvirus", 50),
    ("Lyssavirus", "Lyssavirus", 150),
    ("Enterovirus", "Enterovirus", 300),
    ("Norovirus", "Norovirus", 200),
    ("Rotavirus", "Rotavirus", 200),
    ("Alphavirus", "Alphavirus", 150),
    ("Lentivirus", "Lentivirus", 200),
    ("Phlebovirus", "Phlebovirus", 100),
    ("Spumavirus", "Spumavirus", 50),
    ("Deltaretrovirus", "Deltaretrovirus", 100),
    ("Gammaretrovirus", "Gammaretrovirus", 100),
]

MIN_LEN = 500
MAX_RETRIES = 8
BASE_DELAY = 5


def request_with_retry(url, max_retries=MAX_RETRIES, base_delay=BASE_DELAY, timeout=60):
    """Make HTTP request with exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "glm-locking/1.0 (academic research)",
                "Accept": "application/xml,text/xml,*/*"
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return data.decode("utf-8")
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 500:
                # Server error - retry with longer delay
                pass
            elif e.code == 429:
                # Rate limit - wait longer
                base_delay = max(base_delay, 10)
            elif e.code >= 400:
                if attempt < 3:
                    pass
                else:
                    raise
        except Exception as e:
            last_error = e
        
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt) + (time.time() % 5)
            if attempt % 3 == 0 and attempt > 0:
                print(f"[r{attempt}]", end="", flush=True)
            time.sleep(delay)
    
    raise last_error or Exception("Max retries exceeded")


def search_ncbi(organism, retmax=50000):
    """Search NCBI nucleotide database."""
    params = urllib.parse.urlencode({
        "db": "nucleotide",
        "term": organism,
        "retmax": retmax,
        "usehistory": "y",
    })
    xml_str = request_with_retry(f"{ESEARCH}?{params}")
    
    count_m = re.search(r"<Count>(\d+)</Count>", xml_str)
    webenv_m = re.search(r"<WebEnv>(\S+)</WebEnv>", xml_str)
    qkey_m = re.search(r"<QueryKey>(\d+)</QueryKey>", xml_str)
    
    count = int(count_m.group(1)) if count_m else 0
    webenv = webenv_m.group(1) if webenv_m else ""
    qkey = qkey_m.group(1) if qkey_m else ""
    
    return webenv, qkey, count


def fetch_fasta_batch(webenv, qkey, retstart, retmax):
    """Fetch a batch of FASTA records."""
    params = urllib.parse.urlencode({
        "db": "nucleotide",
        "query_key": qkey,
        "WebEnv": webenv,
        "rettype": "fasta",
        "retmode": "text",
        "retstart": retstart,
        "retmax": retmax,
    })
    return request_with_retry(f"{EFETCH}?{params}", timeout=120)


def parse_fasta(text):
    """Parse FASTA text."""
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


def download_taxon(name, organism, target):
    """Download sequences for one taxon."""
    out_path = os.path.join(OUT_DIR, f"{name}.fasta")
    
    if os.path.exists(out_path):
        n = 0
        with open(out_path) as f:
            for l in f:
                if l.startswith(">"):
                    n += 1
        if n >= target:
            print(f"  [{name}] ✓ {n} (cached)")
            return n
    
    print(f"  [{name}] searching...", end=" ", flush=True)
    
    try:
        webenv, qkey, count = search_ncbi(organism)
    except Exception as e:
        print(f"✗ search failed after retries: {e}")
        return 0
    
    if count == 0:
        print(f"0 results")
        return 0
    
    # Need to overfetch then filter by length
    fetch_total = min(count, target * 10)
    print(f"{count} total", end="", flush=True)
    
    all_fasta = ""
    fetched = 0
    batch_size = 200
    errors = 0
    
    while fetched < fetch_total and errors < 5:
        batch = min(batch_size, fetch_total - fetched)
        try:
            fasta_text = fetch_fasta_batch(webenv, qkey, fetched, batch)
            all_fasta += fasta_text
            fetched += batch
            if fetched % 1000 == 0:
                print(f".{fetched}", end="", flush=True)
            time.sleep(0.25)
        except Exception as e:
            errors += 1
            print(f"!", end="", flush=True)
            time.sleep(10 * errors)
            continue
    
    print(f"={fetched}", end=" ", flush=True)
    
    # Parse, filter, and save
    records = parse_fasta(all_fasta)
    filtered = [(h, s) for h, s in records if len(s) >= MIN_LEN]
    
    # Deduplicate by accession
    seen = set()
    unique = []
    for h, s in filtered:
        acc = h.split()[0]
        if acc not in seen:
            seen.add(acc)
            unique.append((h, s))
    
    unique = unique[:target]
    
    with open(out_path, "w") as f:
        for hdr, seq in unique:
            f.write(f">{hdr}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")
    
    print(f"→ {len(unique)} seqs")
    return len(unique)


def main():
    print("=" * 70)
    print("10k Virus Genome Download — Resilient Edition")
    print(f"Output: {OUT_DIR}")
    print(f"Taxa: {len(TAXA)}")
    print("=" * 70)
    
    total = 0
    for i, (name, organism, target) in enumerate(TAXA):
        n = download_taxon(name, organism, target)
        total += n
        print(f"  [total so far: {total}]")
    
    print()
    print("=" * 70)
    print(f"FINAL TOTAL: {total} sequences")
    print(f"Next: python experiments/exp1_datascale/merge_10k_corpus.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
