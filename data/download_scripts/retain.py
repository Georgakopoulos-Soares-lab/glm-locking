#!/usr/bin/env python3
"""
retain.py  –  Build the SpecDef retain dataset (GTDB bacteria/archaea + IMG/VR phages)

Subcommands (called in order by prepare_retain.sh):
  sample-gtdb      Read GTDB taxonomy TSVs, sample accessions for NCBI download
  extract-contigs  Pick one contig per downloaded genome → gtdb_sampled.fasta
  sample-imgvr     Filter IMG/VR TSV, output sequence IDs for extraction
  build            Combine GTDB + IMG/VR → retain.fasta

Environment variables:
  SCRATCH_DIR  (default: /scratch/10906/arisk/evo_locking_data)
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_BACTERIA     = 5_000
N_ARCHAEA      = 500
MIN_CONTIG_LEN = 10_000   # bp — minimum contig length to keep from GTDB genomes
N_PHAGES       = 3_000
MIN_SEQ_LEN    = 8_000    # bp — minimum IMG/VR sequence length
MIN_USABLE_LEN = 512      # bp — final filter when building retain.fasta

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch/10906/arisk/evo_locking_data")
GTDB_DIR    = os.path.join(SCRATCH_DIR, "gtdb")
IMGVR_DIR   = os.path.join(SCRATCH_DIR, "imgvr")
BUILD_DIR   = os.path.join(SCRATCH_DIR, "retain_build")

# Eukaryotic virus families to exclude from IMG/VR
EXCLUDE_FAMILIES = {
    "Adenoviridae", "Ascoviridae", "Asfarviridae", "Baculoviridae",
    "Herpesviridae", "Iridoviridae", "Marseilleviridae", "Mimiviridae",
    "Nimaviridae", "Nudiviridae", "Papillomaviridae", "Phycodnaviridae",
    "Pithoviridae", "Polyomaviridae", "Poxviridae", "Retroviridae",
    "Reoviridae", "Togaviridae", "Flaviviridae",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def parse_fasta(path: str) -> list[tuple[str, str]]:
    records = []
    header = None
    seq_parts: list[str] = []
    with open(path) as f:
        for line in f:
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


def clean_dna(seq: str) -> str:
    return "".join(c for c in seq.upper().replace("U", "T") if c in "ACGT")


def write_fasta(records: list[tuple[str, str]], path: str, line_width: int = 80):
    with open(path, "w") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), line_width):
                f.write(seq[i : i + line_width] + "\n")


# ---------------------------------------------------------------------------
# sample-gtdb
# ---------------------------------------------------------------------------
def cmd_sample_gtdb():
    output_accs = os.path.join(GTDB_DIR, "sampled_accessions.txt")
    output_tsv  = os.path.join(GTDB_DIR, "sampled_metadata.tsv")

    if os.path.exists(output_accs) and os.path.exists(output_tsv):
        n = sum(1 for _ in open(output_accs))
        print(f"Accession list already exists ({n} genomes), skipping.")
        return

    bac_tsv = os.path.join(GTDB_DIR, "bac120_taxonomy_r220.tsv")
    arc_tsv = os.path.join(GTDB_DIR, "ar53_taxonomy_r220.tsv")
    for p in (bac_tsv, arc_tsv):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found. Run prepare_retain.sh (step 1) first.")

    def parse_taxonomy(path: str) -> dict:
        ranks = ["domain", "phylum", "class", "order", "family", "genus", "species"]
        records: dict = {}
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
                rec = {
                    rank: (lv.split("__", 1)[1] if "__" in lv else lv)
                    for rank, lv in zip(ranks, levels)
                }
                rec["gtdb_accession"] = acc
                rec["ncbi_accession"] = (
                    acc.split("_", 1)[1] if acc.startswith(("RS_", "GB_")) else acc
                )
                records[acc] = rec
        return records

    def sample_across_phyla(records: dict, n_target: int, rng: random.Random) -> list:
        by_phylum: dict = defaultdict(list)
        for rec in records.values():
            if rec.get("phylum"):
                by_phylum[rec["phylum"]].append(rec)
        total = sum(len(v) for v in by_phylum.values())
        sorted_phyla = sorted(by_phylum, key=lambda p: len(by_phylum[p]), reverse=True)
        alloc = {p: max(1, int(len(by_phylum[p]) / total * n_target)) for p in by_phylum}
        current = sum(alloc.values())
        for p in sorted_phyla:
            if current >= n_target:
                break
            add = min(n_target - current, len(by_phylum[p]) - alloc[p])
            if add > 0:
                alloc[p] += add
                current += add
        sampled = []
        for p, recs in by_phylum.items():
            n = min(alloc.get(p, 0), len(recs))
            sampled.extend(rng.sample(recs, n))
        return sampled

    print("Parsing bacterial taxonomy...")
    bac = parse_taxonomy(bac_tsv)
    print(f"  {len(bac):,} species reps")
    print("Parsing archaeal taxonomy...")
    arc = parse_taxonomy(arc_tsv)
    print(f"  {len(arc):,} species reps")

    bac_sampled = sample_across_phyla(bac, N_BACTERIA, random.Random(SEED))
    arc_sampled = sample_across_phyla(arc, N_ARCHAEA, random.Random(SEED + 1))
    all_sampled = bac_sampled + arc_sampled
    print(f"Sampled: {len(bac_sampled)} bacteria + {len(arc_sampled)} archaea = {len(all_sampled)} total")

    os.makedirs(GTDB_DIR, exist_ok=True)
    with open(output_accs, "w") as f:
        for rec in all_sampled:
            f.write(rec["ncbi_accession"] + "\n")
    print(f"Wrote {len(all_sampled)} accessions → {output_accs}")

    cols = ["gtdb_accession", "ncbi_accession", "domain", "phylum", "class", "order",
            "family", "genus", "species"]
    with open(output_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for rec in all_sampled:
            f.write("\t".join(rec.get(c, "") for c in cols) + "\n")


# ---------------------------------------------------------------------------
# extract-contigs
# ---------------------------------------------------------------------------
def cmd_extract_contigs():
    genome_dir   = os.path.join(GTDB_DIR, "genomes")
    output_fasta = os.path.join(BUILD_DIR, "gtdb_sampled.fasta")

    if os.path.exists(output_fasta):
        n = sum(1 for line in open(output_fasta) if line.startswith(">"))
        print(f"GTDB contigs already extracted ({n} sequences), skipping.")
        return

    fna_files = sorted(f for f in os.listdir(genome_dir) if f.endswith(".fna"))
    if not fna_files:
        sys.exit(f"ERROR: No .fna files found in {genome_dir}")

    print(f"Extracting 1 contig ≥ {MIN_CONTIG_LEN:,} bp from {len(fna_files):,} genomes...")
    rng = random.Random(SEED)
    os.makedirs(BUILD_DIR, exist_ok=True)
    written = skipped = 0
    total_bp = 0

    with open(output_fasta, "w") as fout:
        for i, fname in enumerate(fna_files):
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(fna_files)} (written={written}, skipped={skipped})")
            try:
                contigs = parse_fasta(os.path.join(genome_dir, fname))
            except Exception as e:
                print(f"  WARNING: {fname}: {e}")
                skipped += 1
                continue
            valid = [(h, clean_dna(s)) for h, s in contigs if len(s) >= MIN_CONTIG_LEN]
            if not valid:
                skipped += 1
                continue
            header, seq = rng.choice(valid)
            genome_name = fname.replace("_genomic.fna", "")
            fout.write(f">gtdb|{genome_name}|{header}\n")
            for j in range(0, len(seq), 80):
                fout.write(seq[j : j + 80] + "\n")
            written += 1
            total_bp += len(seq)

    print(f"Done. written={written}, skipped={skipped}, total={total_bp / 1e6:.1f} Mb")
    print(f"Output: {output_fasta}")


# ---------------------------------------------------------------------------
# sample-imgvr
# ---------------------------------------------------------------------------
def cmd_sample_imgvr():
    seq_tsv    = os.path.join(IMGVR_DIR, "IMGVR_all_Sequence_information-high_confidence.tsv")
    output_ids = os.path.join(BUILD_DIR, "imgvr_sampled_ids.txt")
    output_tsv = os.path.join(BUILD_DIR, "imgvr_sampled_metadata.tsv")

    if os.path.exists(output_ids) and os.path.exists(output_tsv):
        n = sum(1 for _ in open(output_ids))
        print(f"IMG/VR IDs already sampled ({n} sequences), skipping.")
        return

    if not os.path.exists(seq_tsv):
        sys.exit(f"ERROR: {seq_tsv} not found. Run prepare_retain.sh (step 5) first.")

    print(f"Reading {seq_tsv}...")
    by_votu: dict[str, list[dict]] = defaultdict(list)
    total = passed_len = passed_host = passed_family = 0

    with open(seq_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            seq_id = row.get("UVIG", "").strip()
            if not seq_id:
                continue
            try:
                seq_len = int(row.get("Length", "0"))
            except (ValueError, TypeError):
                continue
            if seq_len < MIN_SEQ_LEN:
                continue
            passed_len += 1

            host_tax = row.get("Host taxonomy prediction", "")
            if not host_tax:
                continue
            host_lower = host_tax.lower()
            if "bacteria" not in host_lower and "archaea" not in host_lower:
                continue
            passed_host += 1

            virus_tax = row.get("Taxonomic classification", "")
            if any(fam.lower() in virus_tax.lower() for fam in EXCLUDE_FAMILIES):
                continue
            passed_family += 1

            votu = row.get("vOTU", seq_id)
            by_votu[votu].append({
                "id": seq_id, "length": seq_len,
                "host": host_tax, "taxonomy": virus_tax, "votu": votu,
            })

    print(
        f"  {total:,} rows → len≥{MIN_SEQ_LEN}: {passed_len:,}"
        f" → prokaryotic host: {passed_host:,}"
        f" → family filter: {passed_family:,}"
        f" → {len(by_votu):,} vOTUs"
    )

    # 1 rep per vOTU (longest sequence)
    votu_reps = [max(recs, key=lambda r: r["length"]) for recs in by_votu.values()]
    rng = random.Random(SEED)
    sampled = votu_reps if len(votu_reps) <= N_PHAGES else rng.sample(votu_reps, N_PHAGES)
    print(f"  Sampled {len(sampled)} vOTU representatives")

    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(output_ids, "w") as f:
        for rec in sampled:
            f.write(rec["id"] + "\n")
    with open(output_tsv, "w") as f:
        f.write("sequence_id\tlength\thost_taxonomy\tvirus_taxonomy\tvotu\n")
        for rec in sampled:
            f.write(f"{rec['id']}\t{rec['length']}\t{rec['host']}\t{rec['taxonomy']}\t{rec['votu']}\n")
    print(f"Wrote {len(sampled)} IDs → {output_ids}")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def cmd_build():
    gtdb_fasta  = os.path.join(BUILD_DIR, "gtdb_sampled.fasta")
    imgvr_fasta = os.path.join(BUILD_DIR, "imgvr_sampled.fasta")
    output      = os.path.join(SCRATCH_DIR, "retain.fasta")

    if os.path.exists(output):
        n = sum(1 for line in open(output) if line.startswith(">"))
        print(f"retain.fasta already built ({n} sequences), skipping.")
        return

    all_records: list[tuple[str, str]] = []

    if os.path.exists(gtdb_fasta):
        gtdb = parse_fasta(gtdb_fasta)
        before = len(all_records)
        for header, seq in gtdb:
            seq = clean_dna(seq)
            if len(seq) >= MIN_USABLE_LEN:
                all_records.append((header, seq))
        print(f"GTDB:   {len(gtdb):,} parsed, {len(all_records) - before:,} kept")
    else:
        print(f"WARNING: {gtdb_fasta} not found")

    if os.path.exists(imgvr_fasta):
        imgvr = parse_fasta(imgvr_fasta)
        before = len(all_records)
        for header, seq in imgvr:
            seq = clean_dna(seq)
            if len(seq) >= MIN_USABLE_LEN:
                if not header.startswith("imgvr|"):
                    header = f"imgvr|{header}"
                all_records.append((header, seq))
        print(f"IMG/VR: {len(imgvr):,} parsed, {len(all_records) - before:,} kept")
    else:
        print(f"WARNING: {imgvr_fasta} not found — proceeding with GTDB only")

    if not all_records:
        sys.exit("ERROR: No sequences found.")

    random.Random(SEED).shuffle(all_records)
    write_fasta(all_records, output)

    total_bp = sum(len(s) for _, s in all_records)
    sources = Counter(
        "GTDB" if h.startswith("gtdb|") else "IMG/VR" if h.startswith("imgvr|") else "other"
        for h, _ in all_records
    )
    print(f"\nretain.fasta: {len(all_records):,} sequences, {total_bp / 1e6:.1f} Mb")
    for src, n in sources.most_common():
        print(f"  {src:10s} {n:,}")

    # Symlink into data/
    proj_data = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    link_path = os.path.join(proj_data, "retain.fasta")
    if os.path.exists(link_path) or os.path.islink(link_path):
        os.remove(link_path)
    os.symlink(output, link_path)
    print(f"Symlinked: {link_path} → {output}")
    print(f"\nDone. Ready for locking: python scripts/lock.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
COMMANDS = {
    "sample-gtdb":     cmd_sample_gtdb,
    "extract-contigs": cmd_extract_contigs,
    "sample-imgvr":    cmd_sample_imgvr,
    "build":           cmd_build,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=COMMANDS, help="Processing step to run")
    args = parser.parse_args()
    COMMANDS[args.command]()
