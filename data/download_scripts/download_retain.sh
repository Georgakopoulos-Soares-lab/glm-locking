#!/usr/bin/env bash
# =============================================================================
# download_retain.sh — Download a representative prokaryotic dataset for SpecDef
#                      retain loss (preserving Evo's pretrained representations)
#
# Downloads ~3,000 complete prokaryotic genomes/contigs from NCBI RefSeq:
#   - ~2,000 diverse bacterial genomes (spread across major phyla)
#   - ~300 archaeal genomes
#   - ~700 bacteriophage genomes (prokaryotic viruses — IN Evo's training)
#
# These are IN-distribution for Evo-1 (GTDB + IMG/VR prokaryotic phages).
# No JGI account needed — uses NCBI efetch only.
#
# Output:  data/retain.fasta  (~200-500 MB, ~3,000 sequences)
#
# Requirements: curl, python3
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$PROJECT_DIR/data/retain.fasta"
TMP_DIR="$(mktemp -d /tmp/retain_dl.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

log() { echo ""; echo ">>> $*"; }

EFETCH="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESEARCH="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# ---------------------------------------------------------------------------
# Search queries — diverse prokaryotic taxa + phages
# Each: label, NCBI query, max seqs
# ---------------------------------------------------------------------------
declare -a QUERIES=(
    # --- Bacteria (major phyla, RefSeq complete genomes) ---
    # Note: "representative genome[filter]" is Assembly-only; use refseq + complete genome for nucleotide
    "Proteobacteria"
    "Pseudomonadota[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "400"

    "Firmicutes"
    "Bacillota[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "300"

    "Actinobacteria"
    "Actinomycetota[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "250"

    "Bacteroidota"
    "Bacteroidota[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "200"

    "Cyanobacteria"
    "Cyanobacteriota[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "100"

    "Spirochaetes"
    "Spirochaetota[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "80"

    "Chloroflexi"
    "Chloroflexota[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "50"

    "Deinococcota"
    "Deinococcota[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "40"

    "Fusobacteria"
    "Fusobacteriota[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "40"

    "Verrucomicrobia"
    "Verrucomicrobiota[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "40"

    "Planctomycetes"
    "Planctomycetota[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "40"

    "Chlamydia"
    "Chlamydiota[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "40"

    "OtherBacteria"
    "Bacteria[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop] NOT Pseudomonadota[Organism] NOT Bacillota[Organism] NOT Actinomycetota[Organism] NOT Bacteroidota[Organism] NOT Cyanobacteriota[Organism] NOT Spirochaetota[Organism]"
    "120"

    # --- Archaea ---
    "Euryarchaeota"
    "Euryarchaeota[Organism] AND refseq[filter] AND complete genome[title]"
    "150"

    "TACK_Archaea"
    "Thermoproteota[Organism] AND refseq[filter] AND complete genome[title]"
    "80"

    "Asgard_Archaea"
    "Asgardarchaeota[Organism] AND refseq[filter] AND complete genome[title]"
    "30"

    "Other_Archaea"
    "Archaea[Organism] AND refseq[filter] AND complete genome[title] AND representative genome[filter] NOT Euryarchaeota[Organism] NOT Thermoproteota[Organism] NOT Asgardarchaeota[Organism]"
    "40"

    # --- Prokaryotic phages (IN Evo's training distribution) ---
    "Caudoviricetes"
    "Caudoviricetes[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"
    "500"

    "Other_phages"
    "bacteriophage[All Fields] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop] NOT Caudoviricetes[Organism]"
    "200"
)

# If retain.fasta exists and is > 50 MB, skip
if [[ -f "$OUT" ]]; then
    SIZE=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    if [[ "$SIZE" -gt 50000000 ]]; then
        N=$(grep -c "^>" "$OUT" 2>/dev/null || echo 0)
        log "retain.fasta already exists ($N sequences, $(du -sh "$OUT" | cut -f1)). Delete to re-download."
        exit 0
    fi
fi

log "Downloading prokaryotic genomes from NCBI RefSeq..."
> "$TMP_DIR/all.fasta"

i=0
TOTAL_SEQS=0
while [[ $i -lt ${#QUERIES[@]} ]]; do
    LABEL="${QUERIES[$i]}"
    QUERY="${QUERIES[$((i+1))]}"
    MAX="${QUERIES[$((i+2))]}"
    i=$((i+3))

    echo ""
    echo "  [$LABEL] Searching (max $MAX)..."

    ENCODED_QUERY=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$QUERY'''))")
    SEARCH_RESULT=$(curl -fsSL --max-time 30 \
        "$ESEARCH?db=nucleotide&term=$ENCODED_QUERY&retmax=$MAX&usehistory=y" \
        2>/dev/null || echo "")

    if [[ -z "$SEARCH_RESULT" ]]; then
        echo "    WARNING: search failed for $LABEL, skipping"
        continue
    fi

    WEBENV=$(echo "$SEARCH_RESULT" | grep -oP '<WebEnv>\K[^<]+' | head -1 || echo "")
    QKEY=$(echo "$SEARCH_RESULT" | grep -oP '<QueryKey>\K[^<]+' | head -1 || echo "")
    COUNT=$(echo "$SEARCH_RESULT" | grep -oP '<Count>\K[^<]+' | head -1 || echo "0")

    if [[ -z "$WEBENV" || -z "$QKEY" ]]; then
        echo "    WARNING: no results for $LABEL"
        continue
    fi

    FETCH_N=$MAX
    if [[ "$COUNT" -lt "$MAX" ]]; then
        FETCH_N=$COUNT
    fi
    echo "    Found $COUNT, fetching $FETCH_N..."

    # Fetch in batches of 50 (bacterial genomes are large, be conservative)
    RETSTART=0
    BATCH=50
    while [[ $RETSTART -lt $FETCH_N ]]; do
        CHUNK_FILE="$TMP_DIR/${LABEL}_${RETSTART}.fasta"
        curl -fsSL --max-time 300 \
            "$EFETCH?db=nucleotide&query_key=$QKEY&WebEnv=$WEBENV&rettype=fasta&retmode=text&retstart=$RETSTART&retmax=$BATCH" \
            -o "$CHUNK_FILE" 2>/dev/null || true
        if [[ -f "$CHUNK_FILE" ]]; then
            cat "$CHUNK_FILE" >> "$TMP_DIR/all.fasta"
            CHUNK_SEQS=$(grep -c "^>" "$CHUNK_FILE" 2>/dev/null || echo 0)
            TOTAL_SEQS=$((TOTAL_SEQS + CHUNK_SEQS))
        fi
        RETSTART=$((RETSTART + BATCH))
        sleep 0.4
    done
    echo "    Done. Running total: $TOTAL_SEQS sequences"
done

# ---------------------------------------------------------------------------
# For bacterial genomes that are multi-Mb, extract a single 50kb contig
# to keep retain.fasta manageable (~200-500 MB instead of 50+ GB)
# ---------------------------------------------------------------------------
log "Processing sequences (extract 50kb window from large genomes, filter min 1kb)..."
python3 -c "
import random
random.seed(42)

records = []
header = None
seq_parts = []
with open('$TMP_DIR/all.fasta') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if header:
                seq = ''.join(seq_parts).upper()
                seq = ''.join(c for c in seq.replace('U','T') if c in 'ACGT')
                if len(seq) >= 1000:
                    records.append((header, seq))
            header = line
            seq_parts = []
        else:
            seq_parts.append(line)
    if header:
        seq = ''.join(seq_parts).upper()
        seq = ''.join(c for c in seq.replace('U','T') if c in 'ACGT')
        if len(seq) >= 1000:
            records.append((header, seq))

print(f'Parsed {len(records)} sequences (>=1kb)')

# For sequences > 100kb, extract a random 50kb window
# This keeps the file manageable while still being representative
MAX_LEN = 50_000
processed = []
for header, seq in records:
    if len(seq) > MAX_LEN * 2:
        start = random.randint(0, len(seq) - MAX_LEN)
        sub = seq[start:start + MAX_LEN]
        new_header = header + f' [window {start}-{start+MAX_LEN} of {len(seq)}]'
        processed.append((new_header, sub))
    else:
        processed.append((header, seq))

with open('$OUT', 'w') as f:
    for header, seq in processed:
        f.write(f'{header}\n')
        for j in range(0, len(seq), 80):
            f.write(seq[j:j+80] + '\n')

total_bp = sum(len(s) for _, s in processed)
lengths = sorted([len(s) for _, s in processed])
print(f'Wrote {len(processed)} sequences to output')
print(f'Total bases: {total_bp:,}')
if lengths:
    print(f'Length range: {lengths[0]:,} - {lengths[-1]:,} bp')
    print(f'Median length: {lengths[len(lengths)//2]:,} bp')
"

log "Done."
echo ""
N=$(grep -c "^>" "$OUT" 2>/dev/null || echo 0)
echo "Final: $N sequences in $OUT ($(du -sh "$OUT" | cut -f1))"
