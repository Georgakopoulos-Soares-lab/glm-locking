# retain.fasta — Download Pipeline

Builds `data/retain.fasta` (~2 GB, ~6,600 sequences) from two public sources,
used as the retain set during SpecDef weight locking.

| Source | Content | Seqs | Download |
|--------|---------|------|----------|
| [GTDB r220](https://gtdb.ecogenomic.org/) | Representative bacteria & archaea | ~5,500 | NCBI via `datasets` CLI |
| [IMG/VR v4](https://img.jgi.doe.gov/vr/) | High-confidence prokaryotic phages | ~3,000 | JGI portal |

**Total downloads:** ~21 MB GTDB metadata + ~15–25 GB sampled genomes + ~200 MB IMG/VR metadata + ~30 GB IMG/VR FASTA (deleted after extraction)

---

## Files

| File | Purpose |
|------|---------|
| `prepare_retain.sh` | Full pipeline: all downloads + calls `retain.py` between steps |
| `retain.py` | All Python processing: sampling, contig extraction, combining |
| `.env` / `.env.example` | JGI credentials (needed for IMG/VR download) |

---

## Requirements

Standard tools available on any Linux/HPC login node — no conda or pip:

```
curl   wget   python3   awk   zcat   xxd   unzip
```

JGI account for IMG/VR: <https://contacts.jgi.doe.gov/registration/new>

---

## Quick Start

```bash
# 1. Set JGI credentials
cp data/download_scripts/.env.example data/download_scripts/.env
#    edit .env: fill in JGI_USER and JGI_PASS

# 2. (Optional) override scratch location — needs ~50 GB free
export SCRATCH_DIR=/path/to/scratch   # default: /scratch/10906/arisk/evo_locking_data

# 3. Run — takes 1–2 h (mostly download time); use screen/tmux
bash data/download_scripts/prepare_retain.sh
```

`data/retain.fasta` will be a symlink to `$SCRATCH_DIR/retain.fasta`.

All steps are **idempotent** — re-running skips completed steps automatically.

---

## What the pipeline does

```
Step 1  download GTDB r220 taxonomy TSVs            (~21 MB)
Step 2  python retain.py sample-gtdb               → sampled_accessions.txt
Step 3  download sampled GTDB genomes via NCBI      (~15–25 GB)
Step 4  python retain.py extract-contigs           → gtdb_sampled.fasta
Step 5  download IMG/VR sequence info TSV           (~200 MB)
Step 6  python retain.py sample-imgvr              → imgvr_sampled_ids.txt
Step 7  download IMG/VR FASTA, extract with awk,
        delete FASTA                                (~30 GB peak, ~50 MB kept)
Step 8  python retain.py build                     → retain.fasta
```

`retain.py` can also be run standalone for any individual step:

```bash
python3 data/download_scripts/retain.py --help
python3 data/download_scripts/retain.py sample-gtdb
```

---

## Sampling parameters

| Parameter | Value |
|-----------|-------|
| GTDB bacteria | 5,000 — proportional across phyla |
| GTDB archaea | 500 — proportional across phyla |
| Min contig length | 10,000 bp |
| IMG/VR phages | up to 3,000 — 1 rep per vOTU (longest) |
| Min IMG/VR length | 8,000 bp |
| Host filter | Bacteria or Archaea only |
| Family exclusions | 19 known eukaryotic virus families |
| Random seed | 42 |

---

## Output structure

```
$SCRATCH_DIR/
├── gtdb/
│   ├── bac120_taxonomy_r220.tsv
│   ├── ar53_taxonomy_r220.tsv
│   ├── sampled_accessions.txt
│   ├── sampled_metadata.tsv
│   └── genomes/                    (~5,500 .fna files, ~18 GB)
├── imgvr/
│   └── IMGVR_all_Sequence_information-high_confidence.tsv
├── retain_build/
│   ├── gtdb_sampled.fasta
│   ├── imgvr_sampled_ids.txt
│   ├── imgvr_sampled_metadata.tsv
│   └── imgvr_sampled.fasta
└── retain.fasta                    (~2 GB — symlinked as data/retain.fasta)
```

---

## Troubleshooting

**JGI download fails / empty file** — check `.env` credentials; JGI tape-backed files re-authenticate on each run automatically.

**GTDB < 80% success rate** — some NCBI accessions are periodically withdrawn; the pipeline warns but continues.

**IMG/VR fewer sequences than expected** — the FASTA is sorted by internal scaffold ID; a small fraction of IDs may not match due to version differences between the TSV and FASTA. The retain set is still valid.

**Disk space** — GTDB genomes (~18 GB) are kept after step 4. The IMG/VR gz (~30 GB) is deleted automatically after extraction.
