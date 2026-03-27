#!/usr/bin/env bash
# =============================================================================
# Step 3: Download only the ~5,500 sampled GTDB genomes from NCBI (~15-25 GB)
#
# Uses the NCBI `datasets` CLI (standalone binary, no conda needed).
# Downloads are done in batches of 500 to avoid timeout/memory issues.
#
# Usage:
#   bash data/download_scripts/03_download_gtdb_genomes.sh
# =============================================================================
set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-/scratch/10906/arisk/evo_locking_data}"
GTDB_DIR="$SCRATCH_DIR/gtdb"
GENOME_DIR="$GTDB_DIR/genomes"
TOOLS_BIN="$SCRATCH_DIR/tools/bin"
ACC_FILE="$GTDB_DIR/sampled_accessions.txt"
BATCH_SIZE=500

if [ ! -f "$ACC_FILE" ]; then
    echo "ERROR: $ACC_FILE not found. Run 02_sample_gtdb_accessions.py first."
    exit 1
fi

mkdir -p "$GENOME_DIR" "$TOOLS_BIN"

# --- Download NCBI datasets CLI if not present ---
DATASETS="$TOOLS_BIN/datasets"
if [ ! -x "$DATASETS" ]; then
    echo "Downloading NCBI datasets CLI..."
    curl -fsSL \
        "https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets" \
        -o "$DATASETS"
    chmod +x "$DATASETS"
    echo "Installed: $DATASETS"
fi

TOTAL=$(wc -l < "$ACC_FILE")
echo "Downloading $TOTAL genomes in batches of $BATCH_SIZE..."
echo "Estimated size: ~15-25 GB total"
echo ""

# Split into batches and download
BATCH_NUM=0
BATCH_ACCS=()

download_batch() {
    local batch_num="$1"
    shift
    local accs=("$@")

    local batch_file="$GENOME_DIR/.batch_${batch_num}.txt"
    local zip_file="$GENOME_DIR/.batch_${batch_num}.zip"

    printf '%s\n' "${accs[@]}" > "$batch_file"

    echo "  Batch ${batch_num}: ${#accs[@]} genomes..."
    "$DATASETS" download genome accession \
        --inputfile "$batch_file" \
        --include genome \
        --filename "$zip_file" \
        --no-progressbar 2>&1 || {
            echo "  WARNING: Batch ${batch_num} had download errors (continuing)"
        }

    if [ -f "$zip_file" ]; then
        local tmp_dir="$GENOME_DIR/.batch_${batch_num}_tmp"
        unzip -o -q "$zip_file" -d "$tmp_dir"
        # NCBI datasets unzips to ncbi_dataset/data/<accession>/*.fna
        find "$tmp_dir" -name "*_genomic.fna" -exec mv {} "$GENOME_DIR/" \;
        rm -rf "$tmp_dir" "$zip_file"
    fi

    rm -f "$batch_file"
}

while IFS= read -r acc; do
    acc=$(echo "$acc" | tr -d '[:space:]')
    [ -z "$acc" ] && continue
    BATCH_ACCS+=("$acc")

    if [ "${#BATCH_ACCS[@]}" -ge "$BATCH_SIZE" ]; then
        BATCH_NUM=$((BATCH_NUM + 1))
        download_batch "$BATCH_NUM" "${BATCH_ACCS[@]}"
        BATCH_ACCS=()
    fi
done < "$ACC_FILE"

# Final partial batch
if [ "${#BATCH_ACCS[@]}" -gt 0 ]; then
    BATCH_NUM=$((BATCH_NUM + 1))
    download_batch "$BATCH_NUM" "${BATCH_ACCS[@]}"
fi

N_DOWNLOADED=$(find "$GENOME_DIR" -maxdepth 1 -name "*.fna" | wc -l)
echo ""
echo "Downloaded $N_DOWNLOADED genome FASTA files to $GENOME_DIR"
echo "Size: $(du -sh "$GENOME_DIR" | cut -f1)"

if [ "$N_DOWNLOADED" -lt "$((TOTAL * 80 / 100))" ]; then
    echo "WARNING: Got $N_DOWNLOADED / $TOTAL (< 80%). Some accessions may be withdrawn or renamed."
fi
