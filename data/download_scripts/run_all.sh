#!/usr/bin/env bash
# =============================================================================
# Master script: build retain.fasta from scratch
#
# Downloads only what's needed (no full 110 GB GTDB tarball, no full IMG/VR FASTA).
# Total download: ~21 MB GTDB metadata + ~15-25 GB selected genomes
#               + ~200 MB IMG/VR metadata (streamed, never storing 30 GB FASTA)
#
# Step-by-step:
#   bash  00_setup_env.sh                  # build seqtk from source
#   bash  01_download_gtdb_metadata.sh     # taxonomy TSVs only (~21 MB)
#   python 02_sample_gtdb_accessions.py    # select 5,500 genomes
#   bash  03_download_gtdb_genomes.sh      # download only those genomes (~15-25 GB)
#   python 04_extract_gtdb_contigs.py      # 1 contig per genome -> fasta
#   bash  05_download_imgvr_metadata.sh    # IMG/VR sequence info TSV only (~200 MB)
#   python 06_sample_imgvr.py              # filter/sample ~3,000 phage IDs
#   bash  07_stream_imgvr_sequences.sh     # stream FASTA, extract sampled seqs (~50-150 MB out)
#   python 08_build_retain.py              # combine -> retain.fasta
#
# Usage:
#   bash data/download_scripts/run_all.sh
#
# Environment variables:
#   SCRATCH_DIR  - where to store all data (default: /scratch/10906/arisk/evo_locking_data)
# =============================================================================
set -euo pipefail

export SCRATCH_DIR="${SCRATCH_DIR:-/scratch/10906/arisk/evo_locking_data}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

echo "============================================"
echo "  evo-locking: Build retain dataset"
echo "============================================"
echo "  Scratch dir: $SCRATCH_DIR"
echo "  Project dir: $PROJECT_DIR"
echo "============================================"
echo ""

mkdir -p "$SCRATCH_DIR"
export PATH="$SCRATCH_DIR/tools/bin:$PATH"

echo ">>> Step 0: Checking prerequisites..."
bash "$SCRIPT_DIR/00_setup_env.sh"
echo ""

echo ">>> Step 1: Downloading GTDB taxonomy (~21 MB)..."
bash "$SCRIPT_DIR/01_download_gtdb_metadata.sh"
echo ""

echo ">>> Step 2: Sampling GTDB accessions..."
python3 "$SCRIPT_DIR/02_sample_gtdb_accessions.py"
echo ""

echo ">>> Step 3: Downloading selected GTDB genomes (~15-25 GB)..."
bash "$SCRIPT_DIR/03_download_gtdb_genomes.sh"
echo ""

echo ">>> Step 4: Extracting contigs..."
python3 "$SCRIPT_DIR/04_extract_gtdb_contigs.py"
echo ""

echo ">>> Step 5: Downloading IMG/VR metadata TSV (~200 MB)..."
bash "$SCRIPT_DIR/05_download_imgvr_metadata.sh"
echo ""

echo ">>> Step 6: Sampling IMG/VR phage IDs..."
python3 "$SCRIPT_DIR/06_sample_imgvr.py"
echo ""

echo ">>> Step 7: Streaming IMG/VR sequences (no full-file storage)..."
bash "$SCRIPT_DIR/07_stream_imgvr_sequences.sh"
echo ""

echo ">>> Step 8: Building retain.fasta..."
python3 "$SCRIPT_DIR/08_build_retain.py"

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  retain.fasta symlinked to: data/retain.fasta"
echo "  Next: python scripts/lock.py  (on a GPU node)"
echo "============================================"
