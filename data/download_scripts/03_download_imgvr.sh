#!/usr/bin/env bash
# =============================================================================
# Step 3: Download IMG/VR v4 high-confidence data from JGI
#
# Downloads:
#   - IMGVR_all_Sequence_information-high_confidence.tsv  (~200 MB)
#   - IMGVR_all_nucleotides-high_confidence.fna.gz        (~30 GB compressed)
#
# Requires JGI credentials in data/download_scripts/.env:
#   JGI_USER="your_email"
#   JGI_PASS="your_password"
#
# Usage:
#   bash data/download_scripts/03_download_imgvr.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/10906/arisk/evo_locking_data}"
IMGVR_DIR="$SCRATCH_DIR/imgvr"

mkdir -p "$IMGVR_DIR"

# --- Load credentials ---
ENV_FILE="$SCRIPT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found."
    echo "Create it with:"
    echo '  JGI_USER="your_email@example.com"'
    echo '  JGI_PASS="your_password"'
    exit 1
fi
source "$ENV_FILE"

if [ -z "${JGI_USER:-}" ] || [ -z "${JGI_PASS:-}" ]; then
    echo "ERROR: JGI_USER and JGI_PASS must be set in $ENV_FILE"
    exit 1
fi

COOKIE_JAR="$IMGVR_DIR/.jgi_cookies"

# --- Login ---
echo "Logging in to JGI as $JGI_USER..."
curl -s 'https://signon.jgi.doe.gov/signon/create' \
    --data-urlencode "login=${JGI_USER}" \
    --data-urlencode "password=${JGI_PASS}" \
    -c "$COOKIE_JAR" > /dev/null

# --- Download sequence metadata TSV ---
SEQ_TSV="$IMGVR_DIR/IMGVR_all_Sequence_information-high_confidence.tsv"
if [ -f "$SEQ_TSV" ]; then
    echo "Sequence info TSV already exists, skipping: $SEQ_TSV"
else
    echo "[1/2] Downloading sequence information TSV (~200 MB)..."
    curl -b "$COOKIE_JAR" --progress-bar \
        -o "$SEQ_TSV" \
        'https://genome.jgi.doe.gov/portal/ext-api/downloads/get_tape_file?blocking=true&url=/IMG_VR/download/_JAMO/63a22c8a3b5d0133c73fb0a4/IMGVR_all_Sequence_information-high_confidence.tsv'
    echo "  Downloaded: $(du -h "$SEQ_TSV" | cut -f1)"
fi

# --- Download nucleotide FASTA ---
NUCLEOTIDE_GZ="$IMGVR_DIR/IMGVR_all_nucleotides-high_confidence.fna.gz"
NUCLEOTIDE_FNA="$IMGVR_DIR/IMGVR_all_nucleotides-high_confidence.fna"
if [ -f "$NUCLEOTIDE_FNA" ] || [ -f "$NUCLEOTIDE_GZ" ]; then
    echo "Nucleotide FASTA already exists, skipping."
else
    echo "[2/2] Downloading nucleotide FASTA (~30 GB compressed)..."
    echo "      This will take a while. Consider running in screen/tmux."
    curl -b "$COOKIE_JAR" --progress-bar \
        -o "$NUCLEOTIDE_GZ" \
        'https://genome.jgi.doe.gov/portal/ext-api/downloads/get_tape_file?blocking=true&url=/IMG_VR/download/_JAMO/63a22c8a3b5d0133c73fb0a0/IMGVR_all_nucleotides-high_confidence.fna.gz'
    echo "  Downloaded: $(du -h "$NUCLEOTIDE_GZ" | cut -f1)"

    echo "Decompressing..."
    gunzip "$NUCLEOTIDE_GZ"
    echo "  Decompressed: $(du -h "$NUCLEOTIDE_FNA" | cut -f1)"
fi

# --- Cleanup ---
rm -f "$COOKIE_JAR"

echo ""
echo "Done. IMG/VR data in: $IMGVR_DIR"
ls -lh "$IMGVR_DIR"
