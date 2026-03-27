#!/usr/bin/env bash
# =============================================================================
# Step 1: Download GTDB r220 taxonomy files only (~21 MB total)
#
# We do NOT download the full 110 GB genome tarball.
# Accessions from these TSVs are used in step 2 to select genomes,
# which are then downloaded individually from NCBI in step 3.
#
# Usage:
#   bash data/download_scripts/01_download_gtdb_metadata.sh
# =============================================================================
set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-/scratch/10906/arisk/evo_locking_data}"
GTDB_DIR="$SCRATCH_DIR/gtdb"

mkdir -p "$GTDB_DIR"

BASE="https://data.gtdb.ecogenomic.org/releases/release220/220.0"

if [ -f "$GTDB_DIR/bac120_taxonomy_r220.tsv" ]; then
    echo "bac120_taxonomy_r220.tsv already exists, skipping."
else
    echo "Downloading bacterial taxonomy (~20 MB)..."
    wget -q --show-progress -O "$GTDB_DIR/bac120_taxonomy_r220.tsv" \
        "${BASE}/bac120_taxonomy_r220.tsv"
fi

if [ -f "$GTDB_DIR/ar53_taxonomy_r220.tsv" ]; then
    echo "ar53_taxonomy_r220.tsv already exists, skipping."
else
    echo "Downloading archaeal taxonomy (~1 MB)..."
    wget -q --show-progress -O "$GTDB_DIR/ar53_taxonomy_r220.tsv" \
        "${BASE}/ar53_taxonomy_r220.tsv"
fi

echo ""
echo "Done."
wc -l "$GTDB_DIR"/*.tsv
