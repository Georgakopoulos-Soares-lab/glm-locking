#!/usr/bin/env bash
# =============================================================================
# prepare_retain.sh  –  Download and assemble the SpecDef retain dataset
#
# Builds data/retain.fasta (~2 GB, ~6,500 sequences) from:
#   - GTDB r220:  ~5,500 representative bacteria and archaea
#   - IMG/VR v4: ~3,000 high-confidence prokaryotic phages
#
# Total downloads: ~21 MB GTDB metadata + ~15-25 GB sampled genomes
#                + ~200 MB IMG/VR metadata + ~30 GB IMG/VR FASTA (deleted after extraction)
#
# Requirements: curl, wget, python3, awk, zcat, xxd, unzip  (standard GNU/Linux)
# JGI account:  https://contacts.jgi.doe.gov/registration/new
# Credentials:  fill in data/download_scripts/.env  (copy from .env.example)
#
# Usage:
#   bash data/download_scripts/prepare_retain.sh
#
# Environment variables:
#   SCRATCH_DIR  (default: /scratch/10906/arisk/evo_locking_data)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export SCRATCH_DIR="${SCRATCH_DIR:-/scratch/10906/arisk/evo_locking_data}"
GTDB_DIR="$SCRATCH_DIR/gtdb"
IMGVR_DIR="$SCRATCH_DIR/imgvr"
BUILD_DIR="$SCRATCH_DIR/retain_build"
TOOLS_BIN="$SCRATCH_DIR/tools/bin"

mkdir -p "$GTDB_DIR" "$IMGVR_DIR" "$BUILD_DIR" "$TOOLS_BIN"

log() { echo ""; echo ">>> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
log "Checking prerequisites..."
missing=0
for cmd in python3 wget curl awk zcat xxd unzip; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "  MISSING: $cmd" >&2; missing=1
    fi
done
[ "$missing" -ne 0 ] && die "Install the missing tools listed above."
echo "  OK"

# Load JGI credentials (needed for steps 5 + 7)
ENV_FILE="$SCRIPT_DIR/.env"
[ -f "$ENV_FILE" ] || die "$ENV_FILE not found. Copy .env.example and fill in your JGI credentials."
source "$ENV_FILE"
[ -z "${JGI_USER:-}" ] && die "JGI_USER not set in $ENV_FILE"
[ -z "${JGI_PASS:-}" ] && die "JGI_PASS not set in $ENV_FILE"

jgi_login() {
    local cookie_jar="$1"
    curl -s -c "$cookie_jar" 'https://signon.jgi.doe.gov/signon/create' \
        --data-urlencode "login=${JGI_USER}" \
        --data-urlencode "password=${JGI_PASS}" > /dev/null
}

# ---------------------------------------------------------------------------
# Step 1: GTDB r220 taxonomy files (~21 MB)
# ---------------------------------------------------------------------------
log "Step 1: Downloading GTDB r220 taxonomy..."
GTDB_BASE="https://data.gtdb.ecogenomic.org/releases/release220/220.0"
for tsv in bac120_taxonomy_r220.tsv ar53_taxonomy_r220.tsv; do
    if [ -f "$GTDB_DIR/$tsv" ]; then
        echo "  exists: $tsv"
    else
        echo "  downloading $tsv..."
        wget -q --show-progress -O "$GTDB_DIR/$tsv" "${GTDB_BASE}/${tsv}"
    fi
done

# ---------------------------------------------------------------------------
# Step 2: Sample GTDB accessions
# ---------------------------------------------------------------------------
log "Step 2: Sampling GTDB accessions..."
python3 "$SCRIPT_DIR/retain.py" sample-gtdb

# ---------------------------------------------------------------------------
# Step 3: Download sampled GTDB genomes (~15-25 GB)
# ---------------------------------------------------------------------------
log "Step 3: Downloading sampled GTDB genomes..."
GENOME_DIR="$GTDB_DIR/genomes"
ACC_FILE="$GTDB_DIR/sampled_accessions.txt"
DATASETS="$TOOLS_BIN/datasets"
BATCH_SIZE=500

mkdir -p "$GENOME_DIR"
TOTAL=$(wc -l < "$ACC_FILE")
N_EXISTING=$(find "$GENOME_DIR" -maxdepth 1 -name "*.fna" 2>/dev/null | wc -l)

if [ "$N_EXISTING" -ge "$((TOTAL * 80 / 100))" ]; then
    echo "  Genomes already present ($N_EXISTING / $TOTAL), skipping."
else
    if [ ! -x "$DATASETS" ]; then
        echo "  Installing NCBI datasets CLI..."
        curl -fsSL "https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets" \
            -o "$DATASETS"
        chmod +x "$DATASETS"
    fi

    echo "  Downloading $TOTAL genomes in batches of $BATCH_SIZE..."
    BATCH_NUM=0
    BATCH_ACCS=()

    download_batch() {
        local n="$1"; shift; local accs=("$@")
        local bf="$GENOME_DIR/.batch_${n}.txt"
        local zf="$GENOME_DIR/.batch_${n}.zip"
        printf '%s\n' "${accs[@]}" > "$bf"
        echo "    batch $n: ${#accs[@]} genomes..."
        "$DATASETS" download genome accession \
            --inputfile "$bf" --include genome --filename "$zf" --no-progressbar 2>&1 \
            || echo "    WARNING: batch $n had download errors (continuing)"
        if [ -f "$zf" ]; then
            local tmp="$GENOME_DIR/.batch_${n}_tmp"
            unzip -o -q "$zf" -d "$tmp"
            find "$tmp" -name "*_genomic.fna" -exec mv {} "$GENOME_DIR/" \;
            rm -rf "$tmp" "$zf"
        fi
        rm -f "$bf"
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

    if [ "${#BATCH_ACCS[@]}" -gt 0 ]; then
        BATCH_NUM=$((BATCH_NUM + 1))
        download_batch "$BATCH_NUM" "${BATCH_ACCS[@]}"
    fi

    N_DOWNLOADED=$(find "$GENOME_DIR" -maxdepth 1 -name "*.fna" | wc -l)
    echo "  Downloaded $N_DOWNLOADED / $TOTAL genome files ($(du -sh "$GENOME_DIR" | cut -f1))"
    if [ "$N_DOWNLOADED" -lt "$((TOTAL * 80 / 100))" ]; then
        echo "  WARNING: < 80% success — some accessions may be withdrawn or renamed at NCBI."
    fi
fi

# ---------------------------------------------------------------------------
# Step 4: Extract one contig per genome
# ---------------------------------------------------------------------------
log "Step 4: Extracting GTDB contigs..."
python3 "$SCRIPT_DIR/retain.py" extract-contigs

# ---------------------------------------------------------------------------
# Step 5: IMG/VR sequence metadata (~200 MB)
# ---------------------------------------------------------------------------
log "Step 5: Downloading IMG/VR metadata..."
SEQ_TSV="$IMGVR_DIR/IMGVR_all_Sequence_information-high_confidence.tsv"

if [ -f "$SEQ_TSV" ]; then
    echo "  exists: $SEQ_TSV"
else
    COOKIE_JAR="$IMGVR_DIR/.jgi_cookies"
    echo "  Logging in to JGI as $JGI_USER..."
    jgi_login "$COOKIE_JAR"
    echo "  Downloading sequence info TSV (~200 MB)..."
    curl -b "$COOKIE_JAR" --progress-bar -o "$SEQ_TSV" \
        'https://genome.jgi.doe.gov/portal/ext-api/downloads/get_tape_file?blocking=true&url=/IMG_VR/download/_JAMO/63a22c8a3b5d0133c73fb0a4/IMGVR_all_Sequence_information-high_confidence.tsv'
    rm -f "$COOKIE_JAR"
    echo "  $(du -h "$SEQ_TSV" | cut -f1)"
fi

# ---------------------------------------------------------------------------
# Step 6: Sample IMG/VR phage IDs
# ---------------------------------------------------------------------------
log "Step 6: Sampling IMG/VR phage IDs..."
python3 "$SCRIPT_DIR/retain.py" sample-imgvr

# ---------------------------------------------------------------------------
# Step 7: Download IMG/VR FASTA, extract sampled sequences, delete FASTA
# ---------------------------------------------------------------------------
log "Step 7: Extracting IMG/VR sequences..."
IDS_FILE="$BUILD_DIR/imgvr_sampled_ids.txt"
IMGVR_FASTA="$BUILD_DIR/imgvr_sampled.fasta"
NUCLEOTIDE_GZ="$IMGVR_DIR/IMGVR_all_nucleotides-high_confidence.fna.gz"
NUCLEOTIDE_URL='https://genome.jgi.doe.gov/portal/ext-api/downloads/get_tape_file?blocking=true&url=/IMG_VR/download/_JAMO/63a22c8a3b5d0133c73fb0a0/IMGVR_all_nucleotides-high_confidence.fna.gz'

if [ -f "$IMGVR_FASTA" ]; then
    echo "  exists: $IMGVR_FASTA ($(grep -c '^>' "$IMGVR_FASTA") sequences)"
else
    N_IDS=$(wc -l < "$IDS_FILE")

    if [ ! -f "$NUCLEOTIDE_GZ" ]; then
        FREE_KB=$(df -k "$SCRATCH_DIR" | awk 'NR==2 {print $4}')
        NEED_KB=$((32 * 1024 * 1024))
        if [ "$FREE_KB" -lt "$NEED_KB" ]; then
            die "Not enough space in $SCRATCH_DIR: $((FREE_KB / 1024 / 1024)) GB free, need ~32 GB."
        fi
        COOKIE_JAR="$IMGVR_DIR/.jgi_cookies"
        echo "  Logging in to JGI..."
        jgi_login "$COOKIE_JAR"
        echo "  Downloading nucleotide FASTA (~30 GB). Run in screen/tmux."
        curl -L --fail --progress-bar -b "$COOKIE_JAR" -o "$NUCLEOTIDE_GZ" "$NUCLEOTIDE_URL"
        rm -f "$COOKIE_JAR"
    fi

    # Verify gzip magic bytes
    FIRST2=$(head -c 2 "$NUCLEOTIDE_GZ" | xxd -p)
    if [ "$FIRST2" != "1f8b" ]; then
        echo "ERROR: Not a valid gzip file (magic bytes: $FIRST2). Check JGI credentials."
        head -c 500 "$NUCLEOTIDE_GZ"; echo ""
        rm -f "$NUCLEOTIDE_GZ"; exit 1
    fi

    # Extract only the sampled sequences.
    # IMG/VR headers: >IMGVR_UViG_XXX|scaffold_id|...|coords — match on UVIG prefix (before |).
    echo "  Extracting $N_IDS sequences..."
    zcat "$NUCLEOTIDE_GZ" | awk -v ids_file="$IDS_FILE" '
        BEGIN { while ((getline id < ids_file) > 0) ids[id] = 1 }
        /^>/ { name = substr($1, 2); split(name, a, "|"); keep = (a[1] in ids) }
        keep { print }
    ' > "$IMGVR_FASTA"

    rm -f "$NUCLEOTIDE_GZ"

    if [ ! -s "$IMGVR_FASTA" ]; then
        rm -f "$IMGVR_FASTA"
        die "Extraction produced empty output."
    fi

    N_EXT=$(grep -c '^>' "$IMGVR_FASTA")
    echo "  Extracted $N_EXT / $N_IDS sequences ($(du -h "$IMGVR_FASTA" | cut -f1))"
    if [ "$N_EXT" -lt "$((N_IDS / 2))" ]; then
        echo "  WARNING: < 50% found — FASTA may be incomplete or IDs have a versioning mismatch."
    fi
fi

# ---------------------------------------------------------------------------
# Step 8: Combine into retain.fasta
# ---------------------------------------------------------------------------
log "Step 8: Building retain.fasta..."
python3 "$SCRIPT_DIR/retain.py" build

echo ""
echo "============================================"
echo "  Done!  $PROJECT_DIR/data/retain.fasta"
echo "============================================"
