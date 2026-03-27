#!/usr/bin/env bash
# =============================================================================
# Step 1: Download GTDB r220 taxonomy + representative genomes
#
# Downloads:
#   - bac120_taxonomy_r220.tsv   (~20 MB)
#   - ar53_taxonomy_r220.tsv     (~1 MB)
#   - gtdb_genomes_reps_r220.tar.gz  (~110 GB, all species-rep genomes)
#
# This is the full representative genome set. We sample from it in step 2.
# The tarball is large but it's the canonical source matching Evo's training.
#
# Usage:
#   bash data/download_scripts/01_download_gtdb.sh
# =============================================================================
set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-/scratch/10906/arisk/evo_locking_data}"
GTDB_DIR="$SCRATCH_DIR/gtdb"

mkdir -p "$GTDB_DIR"

GTDB_BASE="https://data.gtdb.ecogenomic.org/releases/release220/220.0"

# --- Taxonomy files (small, fast) ---
echo "[1/3] Downloading bacterial taxonomy..."
wget -q --show-progress -O "$GTDB_DIR/bac120_taxonomy_r220.tsv" \
    "${GTDB_BASE}/bac120_taxonomy_r220.tsv"

echo "[2/3] Downloading archaeal taxonomy..."
wget -q --show-progress -O "$GTDB_DIR/ar53_taxonomy_r220.tsv" \
    "${GTDB_BASE}/ar53_taxonomy_r220.tsv"

echo ""
echo "Taxonomy files:"
wc -l "$GTDB_DIR"/*.tsv

# --- Representative genomes tarball ---
TARBALL="$GTDB_DIR/gtdb_genomes_reps_r220.tar.gz"
GENOME_DIR="$GTDB_DIR/gtdb_genomes_reps_r220"

if [ -d "$GENOME_DIR" ] && [ "$(find "$GENOME_DIR" -name '*.fna.gz' 2>/dev/null | head -1)" != "" ]; then
    echo ""
    echo "Genome directory already exists: $GENOME_DIR"
    echo "Skipping download. Delete it to re-download."
else
    echo ""
    echo "[3/3] Downloading representative genomes (~110 GB)..."
    echo "      This will take a while. Consider running in a screen/tmux session."
    wget -q --show-progress -O "$TARBALL" \
        "${GTDB_BASE}/genomic_files_reps/gtdb_genomes_reps_r220.tar.gz"

    echo "Extracting tarball..."
    mkdir -p "$GENOME_DIR"
    tar -xzf "$TARBALL" -C "$GTDB_DIR"

    echo "Removing tarball to save space..."
    rm -f "$TARBALL"
fi

echo ""
echo "Done. GTDB data in: $GTDB_DIR"
echo "Genome count:"
find "$GENOME_DIR" -name '*.fna.gz' 2>/dev/null | wc -l
