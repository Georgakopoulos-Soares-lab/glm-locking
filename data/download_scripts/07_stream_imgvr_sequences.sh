#!/usr/bin/env bash
# =============================================================================
# Step 7: Extract sampled IMG/VR sequences via streaming (no full-file storage)
#
# Streams the nucleotide FASTA (~30 GB compressed) from JGI on-the-fly:
#   curl (gzipped stream) | zcat | seqtk subseq - <ids> > output.fasta
#
# This extracts only the ~3,000 sampled sequences without storing the full
# 30 GB FASTA on disk. Final output is ~50-150 MB.
#
# Requires:
#   - JGI credentials in data/download_scripts/.env
#   - seqtk on PATH (built by 00_setup_env.sh)
#   - $SCRATCH_DIR/retain_build/imgvr_sampled_ids.txt (from 06_sample_imgvr.py)
#
# Outputs:
#   $SCRATCH_DIR/retain_build/imgvr_sampled.fasta
#
# Usage:
#   bash data/download_scripts/07_stream_imgvr_sequences.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/10906/arisk/evo_locking_data}"
IDS_FILE="$SCRATCH_DIR/retain_build/imgvr_sampled_ids.txt"
OUTPUT_FASTA="$SCRATCH_DIR/retain_build/imgvr_sampled.fasta"

# --- Checks ---
if [ ! -f "$IDS_FILE" ]; then
    echo "ERROR: ID list not found: $IDS_FILE"
    echo "       Run 06_sample_imgvr.py first."
    exit 1
fi

N_IDS=$(wc -l < "$IDS_FILE")
echo "Found $N_IDS sequence IDs to extract."

if command -v seqtk &>/dev/null; then
    echo "seqtk: $(which seqtk)"
else
    echo "ERROR: seqtk not found. Run 00_setup_env.sh first."
    exit 1
fi

if [ -f "$OUTPUT_FASTA" ]; then
    N_SEQS=$(grep -c '^>' "$OUTPUT_FASTA" || true)
    echo "Output already exists with $N_SEQS sequences: $OUTPUT_FASTA"
    echo "Delete it to re-extract."
    exit 0
fi

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

COOKIE_JAR="$SCRATCH_DIR/imgvr/.jgi_stream_cookies"
mkdir -p "$SCRATCH_DIR/imgvr"

# --- Login ---
echo "Logging in to JGI as $JGI_USER..."
curl -s 'https://signon.jgi.doe.gov/signon/create' \
    --data-urlencode "login=${JGI_USER}" \
    --data-urlencode "password=${JGI_PASS}" \
    -c "$COOKIE_JAR" > /dev/null

# --- Stream + extract ---
# Pipe: JGI gzipped stream -> zcat (decompress) -> seqtk subseq (filter by ID)
# The full ~30 GB never lands on disk; only the matched sequences are written.
NUCLEOTIDE_URL='https://genome.jgi.doe.gov/portal/ext-api/downloads/get_tape_file?blocking=true&url=/IMG_VR/download/_JAMO/63a22c8a3b5d0133c73fb0a0/IMGVR_all_nucleotides-high_confidence.fna.gz'

echo "Streaming nucleotide FASTA from JGI (this will take a while)..."
echo "  Extracting $N_IDS sequences into $OUTPUT_FASTA"

curl -s -b "$COOKIE_JAR" "$NUCLEOTIDE_URL" \
    | zcat \
    | seqtk subseq - "$IDS_FILE" \
    > "$OUTPUT_FASTA"

# --- Cleanup ---
rm -f "$COOKIE_JAR"

# --- Verify ---
if [ ! -s "$OUTPUT_FASTA" ]; then
    echo "ERROR: Output FASTA is empty. Check credentials and network."
    rm -f "$OUTPUT_FASTA"
    exit 1
fi

N_EXTRACTED=$(grep -c '^>' "$OUTPUT_FASTA")
SIZE=$(du -h "$OUTPUT_FASTA" | cut -f1)
echo ""
echo "Extracted $N_EXTRACTED sequences ($SIZE) to $OUTPUT_FASTA"

if [ "$N_EXTRACTED" -lt "$((N_IDS / 2))" ]; then
    echo "WARNING: Only $N_EXTRACTED of $N_IDS expected sequences were found."
    echo "         The FASTA may be incomplete or IDs may not match."
fi

echo "Step 7 complete. Next: python3 08_build_retain.py"
