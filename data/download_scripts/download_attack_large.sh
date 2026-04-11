#!/usr/bin/env bash
# =============================================================================
# download_attack_large.sh — Download a proper-sized eukaryotic virus dataset
#
# Downloads ~500 complete eukaryotic virus genomes from NCBI RefSeq.
# These are human/animal viruses from families explicitly EXCLUDED from
# Evo-1's OpenGenome pretraining, making them genuinely OOD.
#
# Output:  data/attack.fasta  (~60-80 MB, ~500 sequences)
#
# Strategy: Use NCBI esearch + efetch to pull complete viral genomes from
# families excluded by the OpenGenome safety filter.
#
# Requirements: curl
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$PROJECT_DIR/data/attack.fasta"
TMP_DIR="$(mktemp -d /tmp/attack_dl.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

log() { echo ""; echo ">>> $*"; }

EFETCH="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESEARCH="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# ---------------------------------------------------------------------------
# Define virus families to download (all excluded from OpenGenome)
# Each: family name, NCBI search query, max sequences to fetch
# We target ~500 total sequences across diverse families
# ---------------------------------------------------------------------------
declare -a QUERIES=(
    # Family                    Search query (RefSeq complete genomes)           Max
    "Herpesviridae"             "Herpesviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"  "80"
    "Poxviridae"                "Poxviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"     "60"
    "Adenoviridae"              "Adenoviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"   "60"
    "Retroviridae"              "Retroviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"   "50"
    "Coronaviridae"             "Coronaviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"  "50"
    "Flaviviridae"              "Flaviviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"   "50"
    "Paramyxoviridae"           "Paramyxoviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]" "40"
    "Filoviridae"               "Filoviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"    "30"
    "Reoviridae"                "Reoviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"     "30"
    "Caliciviridae"             "Caliciviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"  "30"
    "Papillomaviridae"          "Papillomaviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]" "30"
    "Rhabdoviridae"             "Rhabdoviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]"  "30"
    "Orthomyxoviridae"          "Orthomyxoviridae[Organism] AND refseq[filter] AND complete genome[title] AND biomol_genomic[prop]" "20"
)

# If attack.fasta already exists and is >10MB, skip
if [[ -f "$OUT" ]]; then
    SIZE=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    if [[ "$SIZE" -gt 10000000 ]]; then
        N=$(grep -c "^>" "$OUT" 2>/dev/null || echo 0)
        log "attack.fasta already exists ($N sequences, $(du -sh "$OUT" | cut -f1)). Delete to re-download."
        exit 0
    fi
fi

log "Downloading eukaryotic virus genomes from NCBI RefSeq..."
> "$TMP_DIR/all.fasta"

i=0
TOTAL_SEQS=0
while [[ $i -lt ${#QUERIES[@]} ]]; do
    FAMILY="${QUERIES[$i]}"
    QUERY="${QUERIES[$((i+1))]}"
    MAX="${QUERIES[$((i+2))]}"
    i=$((i+3))

    echo ""
    echo "  [$FAMILY] Searching (max $MAX)..."

    # Search for IDs
    SEARCH_RESULT=$(curl -fsSL --max-time 30 \
        "$ESEARCH?db=nucleotide&term=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$QUERY'))")&retmax=$MAX&usehistory=y" \
        2>/dev/null || echo "")

    if [[ -z "$SEARCH_RESULT" ]]; then
        echo "    WARNING: search failed for $FAMILY, skipping"
        continue
    fi

    # Extract WebEnv and QueryKey for efetch
    WEBENV=$(echo "$SEARCH_RESULT" | grep -oP '<WebEnv>\K[^<]+' | head -1 || echo "")
    QKEY=$(echo "$SEARCH_RESULT" | grep -oP '<QueryKey>\K[^<]+' | head -1 || echo "")
    COUNT=$(echo "$SEARCH_RESULT" | grep -oP '<Count>\K[^<]+' | head -1 || echo "0")

    if [[ -z "$WEBENV" || -z "$QKEY" ]]; then
        echo "    WARNING: no results for $FAMILY"
        continue
    fi

    FETCH_N=$MAX
    if [[ "$COUNT" -lt "$MAX" ]]; then
        FETCH_N=$COUNT
    fi
    echo "    Found $COUNT, fetching $FETCH_N..."

    # Fetch FASTA in batches of 100 (NCBI limit)
    RETSTART=0
    BATCH=100
    while [[ $RETSTART -lt $FETCH_N ]]; do
        CHUNK_FILE="$TMP_DIR/${FAMILY}_${RETSTART}.fasta"
        curl -fsSL --max-time 120 \
            "$EFETCH?db=nucleotide&query_key=$QKEY&WebEnv=$WEBENV&rettype=fasta&retmode=text&retstart=$RETSTART&retmax=$BATCH" \
            -o "$CHUNK_FILE" 2>/dev/null || true
        if [[ -f "$CHUNK_FILE" ]]; then
            cat "$CHUNK_FILE" >> "$TMP_DIR/all.fasta"
            CHUNK_SEQS=$(grep -c "^>" "$CHUNK_FILE" 2>/dev/null || echo 0)
            TOTAL_SEQS=$((TOTAL_SEQS + CHUNK_SEQS))
        fi
        RETSTART=$((RETSTART + BATCH))
        sleep 0.4  # NCBI rate limit
    done
    echo "    Done. Running total: $TOTAL_SEQS sequences"
done

# Filter: remove empty sequences and those <1kb
log "Filtering sequences (min 1 kb)..."
python3 -c "
import sys
records = []
header = None
seq_parts = []
with open('$TMP_DIR/all.fasta') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if header and len(''.join(seq_parts)) >= 1000:
                records.append((header, ''.join(seq_parts)))
            header = line
            seq_parts = []
        else:
            seq_parts.append(line.upper())
    if header and len(''.join(seq_parts)) >= 1000:
        records.append((header, ''.join(seq_parts)))

with open('$OUT', 'w') as f:
    for header, seq in records:
        f.write(f'{header}\n')
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + '\n')

print(f'Wrote {len(records)} sequences to $OUT')
total_bp = sum(len(s) for _, s in records)
print(f'Total bases: {total_bp:,}')
lengths = sorted([len(s) for _, s in records])
if lengths:
    print(f'Length range: {lengths[0]:,} - {lengths[-1]:,} bp')
    print(f'Median length: {lengths[len(lengths)//2]:,} bp')
"

log "Done."
echo ""
N=$(grep -c "^>" "$OUT" 2>/dev/null || echo 0)
echo "Final: $N sequences in $OUT ($(du -sh "$OUT" | cut -f1))"
