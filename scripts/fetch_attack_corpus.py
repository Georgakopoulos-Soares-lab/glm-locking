#!/usr/bin/env python3
"""Rebuild the 910-genome attack corpus from its accession manifest.

WHICH GENOMES ARE IN THE CORPUS
-------------------------------
The corpus is *defined by* ``data/attack_accessions.txt`` — 910 NCBI RefSeq
accessions (896 ``NC_*``, 14 ``AC_*``), one per line. That file, not the
original Entrez queries, is the authoritative definition.

How that list was originally derived (see
``data/download_scripts/download_attack_large.sh``): complete eukaryotic-virus
reference genomes were pulled from NCBI Nucleotide with ``refseq[filter]`` via
the E-utilities ``esearch``/``efetch`` API on 2026-04-06, across 13 virus
families absent from Evo-1's OpenGenome pretraining corpus — Herpesviridae,
Poxviridae, Adenoviridae, Retroviridae, Coronaviridae, Flaviviridae,
Paramyxoviridae, Filoviridae, Reoviridae, Caliciviridae, Papillomaviridae,
Rhabdoviridae, Orthomyxoviridae — then filtered to >= 1 kb.

The original queries are deliberately NOT re-run here. NCBI RefSeq contents
drift (records are added, revised and suppressed), so re-issuing a taxonomic
search returns a different set over time. Pinning the accessions makes the
corpus reproducible; re-running the search would not.

Selection criterion, stated plainly: the corpus is selected for *absence from
Evo-1 pretraining*, not for human pathogenicity. OpenGenome comprises only GTDB
prokaryotic genomes, IMG/PR plasmids and prokaryotic viruses, so all 910
eukaryotic viruses are out of distribution at the genome level. Roughly 14% of
the corpus is human-infecting; the remainder infects other vertebrates, plants,
insects, fungi or protists. See ``data/attack_accessions.tsv`` for per-accession
organism descriptions and train/held-out assignment.

USAGE
-----
    # verify an existing corpus against the manifest (no network)
    python scripts/fetch_attack_corpus.py --verify-only

    # rebuild data/attack.fasta from NCBI, then regenerate the split
    python scripts/fetch_attack_corpus.py --split

Re-fetching yields the same 910 records but in manifest order rather than the
original family-by-family download order. This does not affect the split:
``scripts/split_attack_fasta.py`` groups by genus, sorts the genus keys and
shuffles under a fixed seed, so its output depends on the *set* of records, not
their order.

NCBI asks that you identify yourself. Set NCBI_EMAIL (and optionally
NCBI_API_KEY, which raises the rate limit from 3 to 10 requests/second).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.parse
import urllib.request

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DEFAULT_MANIFEST = "data/attack_accessions.txt"
DEFAULT_OUT = "data/attack.fasta"
WRAP = 80
MIN_LEN = 1000


def read_manifest(path: str) -> list[str]:
    with open(path) as f:
        accs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if len(accs) != len(set(accs)):
        raise SystemExit(f"{path}: contains duplicate accessions")
    return accs


def parse_fasta(path: str):
    """Yield (header, sequence) pairs."""
    header, chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def fetch_batch(accs: list[str], email: str, api_key: str, retries: int = 4) -> str:
    params = {
        "db": "nuccore",
        "id": ",".join(accs),
        "rettype": "fasta",
        "retmode": "text",
        "tool": "glm-locking",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    url = f"{EFETCH}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            wait = 3 * (attempt + 1)
            print(f"    retry {attempt + 1}/{retries} after {exc} (sleep {wait}s)",
                  file=sys.stderr)
            time.sleep(wait)
    raise SystemExit(f"efetch failed for batch starting {accs[0]}")


def verify(fasta_path: str, manifest: list[str]) -> int:
    """Compare a FASTA against the manifest. Returns a process exit code."""
    if not os.path.exists(fasta_path):
        print(f"MISSING: {fasta_path}", file=sys.stderr)
        return 1
    got = [h.split()[0] for h, _ in parse_fasta(fasta_path)]
    want_set, got_set = set(manifest), set(got)
    missing, extra = sorted(want_set - got_set), sorted(got_set - want_set)
    short = [h.split()[0] for h, s in parse_fasta(fasta_path) if len(s) < MIN_LEN]

    print(f"manifest : {len(manifest)} accessions")
    print(f"fasta    : {len(got)} records ({len(got_set)} unique)")
    if missing:
        print(f"MISSING from fasta ({len(missing)}): {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")
    if extra:
        print(f"EXTRA in fasta ({len(extra)}): {extra[:10]}"
              f"{' ...' if len(extra) > 10 else ''}")
    if short:
        print(f"WARNING: {len(short)} records under {MIN_LEN} bp: {short[:10]}")
    if not missing and not extra:
        print("OK — fasta matches the manifest exactly")
        return 0
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST,
                   help=f"accession list (default: {DEFAULT_MANIFEST})")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"output FASTA (default: {DEFAULT_OUT})")
    p.add_argument("--batch-size", type=int, default=100,
                   help="accessions per efetch request (NCBI max ~200)")
    p.add_argument("--verify-only", action="store_true",
                   help="check an existing FASTA against the manifest; no network")
    p.add_argument("--limit", type=int, default=0,
                   help="fetch only the first N accessions (smoke test)")
    p.add_argument("--split", action="store_true",
                   help="after fetching, regenerate the train/held-out split")
    p.add_argument("--email", default=os.environ.get("NCBI_EMAIL", ""))
    p.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY", ""))
    args = p.parse_args()

    manifest = read_manifest(args.manifest)

    if args.verify_only:
        return verify(args.out, manifest)

    accs = manifest[:args.limit] if args.limit else manifest
    if os.path.exists(args.out) and not args.limit:
        print(f"{args.out} already exists — refusing to overwrite. "
              f"Delete it first, or use --verify-only.", file=sys.stderr)
        return 1

    # NCBI: 3 req/s without a key, 10 with one.
    delay = 0.15 if args.api_key else 0.4
    if not args.email:
        print("NOTE: set NCBI_EMAIL so NCBI can contact you about heavy usage.",
              file=sys.stderr)

    kept = dropped = 0
    with open(args.out, "w") as out:
        for i in range(0, len(accs), args.batch_size):
            chunk = accs[i:i + args.batch_size]
            print(f"  fetching {i + 1}-{i + len(chunk)} of {len(accs)}")
            text = fetch_batch(chunk, args.email, args.api_key)
            tmp = f"{args.out}.batch.tmp"
            with open(tmp, "w") as t:
                t.write(text)
            for hdr, seq in parse_fasta(tmp):
                seq = seq.upper()
                if len(seq) < MIN_LEN:
                    dropped += 1
                    continue
                out.write(">" + hdr + "\n")
                for j in range(0, len(seq), WRAP):
                    out.write(seq[j:j + WRAP] + "\n")
                kept += 1
            os.remove(tmp)
            time.sleep(delay)

    print(f"\nwrote {kept} records to {args.out}"
          f"{f' ({dropped} dropped under {MIN_LEN} bp)' if dropped else ''}")

    if args.limit:
        print("(--limit set: skipping manifest verification and split)")
        return 0

    rc = verify(args.out, manifest)
    if rc != 0:
        print("Corpus does not match the manifest — not splitting.", file=sys.stderr)
        return rc

    if args.split:
        print("\nregenerating genus-disjoint split...")
        here = os.path.dirname(os.path.abspath(__file__))
        sys.argv = ["split_attack_fasta.py", "--input", args.out]
        sys.path.insert(0, here)
        import split_attack_fasta  # noqa: PLC0415 - imported for its main()
        split_attack_fasta.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
