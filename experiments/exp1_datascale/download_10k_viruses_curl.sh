#!/usr/bin/env bash
# =============================================================================
# download_10k_viruses_curl.sh — Download ~10,000 viral genomes via NCBI eutils
#
# Uses curl to hit NCBI's esearch + efetch directly, same proven approach as
# download_attack_large.sh but targeting many more families and sequences.
#
# Output: experiments/exp1_datascale/downloads/<family>.fasta
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$PROJECT_DIR/experiments/exp1_datascale/downloads"
TMP_DIR="$(mktemp -d /tmp/virus10k_dl.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

ESEARCH="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

mkdir -p "$OUT_DIR"

log() { echo ""; echo ">>> $(date '+%H:%M:%S') $*"; }

# ---------------------------------------------------------------------------
# Define families to download
# Format: family_name | NCBI query | max sequences
# We target ~10,000 total (minus the existing 910)
# Trying queries WITHOUT refseq[filter] to avoid GI-to-accession issues
# ---------------------------------------------------------------------------
declare -a QUERIES=(
    # Family                 Query                                                                                    Max
    "Herpesviridae"          "Herpesviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"            600
    "Poxviridae"             "Poxviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"               400
    "Adenoviridae"           "Adenoviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"             400
    "Retroviridae_HIV"       "Human immunodeficiency virus[Organism] AND complete genome[title] AND biomol_genomic[prop]" 500
    "Retroviridae_other"     "Retroviridae[Organism] NOT HIV[Organism] AND complete genome[title] AND biomol_genomic[prop]" 200
    "Coronaviridae_SARS2"    "Severe acute respiratory syndrome coronavirus 2[Organism] AND complete genome[title] AND biomol_genomic[prop]" 500
    "Coronaviridae_other"    "Coronaviridae[Organism] NOT SARS-CoV-2[Organism] AND complete genome[title] AND biomol_genomic[prop]" 300
    "Flaviviridae_Dengue"    "Dengue virus[Organism] AND complete genome[title] AND biomol_genomic[prop]"             400
    "Flaviviridae_Zika"      "Zika virus[Organism] AND complete genome[title] AND biomol_genomic[prop]"               300
    "Flaviviridae_WNV"       "West Nile virus[Organism] AND complete genome[title] AND biomol_genomic[prop]"          300
    "Flaviviridae_HCV"       "Hepatitis C virus[Organism] AND complete genome[title] AND biomol_genomic[prop]"        500
    "Flaviviridae_YFV"       "Yellow fever virus[Organism] AND complete genome[title] AND biomol_genomic[prop]"       100
    "Flaviviridae_other"     "Flaviviridae[Organism] NOT Dengue[Organism] NOT Zika[Organism] NOT \"West Nile\"[Organism] NOT \"Hepatitis C\"[Organism] NOT \"Yellow fever\"[Organism] AND complete genome[title] AND biomol_genomic[prop]" 200
    "Paramyxoviridae"        "Paramyxoviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"          400
    "Filoviridae"            "Filoviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"              200
    "Reoviridae"             "Reoviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"               300
    "Caliciviridae"          "Caliciviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"            200
    "Papillomaviridae"       "Papillomaviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"         500
    "Rhabdoviridae"          "Rhabdoviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"            200
    "Orthomyxoviridae_InfA"  "Influenza A virus[Organism] AND complete genome[title] AND biomol_genomic[prop]"        600
    "Orthomyxoviridae_other" "Orthomyxoviridae[Organism] NOT \"Influenza A\"[Organism] AND complete genome[title] AND biomol_genomic[prop]" 200
    "Picornaviridae"         "Picornaviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"           300
    "Hepadnaviridae"         "Hepadnaviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"           200
    "Polyomaviridae"         "Polyomaviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"           200
    "Togaviridae"            "Togaviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"              200
    "Bunyaviridae"           "Bunyaviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"             200
    "Astroviridae"           "Astroviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"             150
    "Hantaviridae"           "Hantaviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"             150
    "Arenaviridae"           "Arenaviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"             150
    "Nairoviridae"           "Nairoviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"             100
    "Peribunyaviridae"       "Peribunyaviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"         100
    "Phenuiviridae"          "Phenuiviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"            100
    "Parvoviridae"           "Parvoviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"             200
    "Anelloviridae"          "Anelloviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"            200
    "Circoviridae"           "Circoviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"             150
    "Birnaviridae"           "Birnaviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"             100
    "Iridoviridae"           "Iridoviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"             100
    "Asfarviridae"           "Asfarviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"             50
    "Mimiviridae"            "Mimiviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"              50
    "Baculoviridae"          "Baculoviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"            100
    "Nimaviridae"            "Nimaviridae[Organism] AND complete genome[title] AND biomol_genomic[prop]"              50
)

log "Downloading ~10k viral genomes from NCBI..."

i=0
TOTAL_DL=0
while [[ $i -lt ${#QUERIES[@]} ]]; do
    FAMILY="${QUERIES[$i]}"
    QUERY="${QUERIES[$((i+1))]}"
    MAX="${QUERIES[$((i+2))]}"
    i=$((i+3))

    OUTFILE="$OUT_DIR/${FAMILY}.fasta"
    
    # Skip if already have enough
    if [[ -f "$OUTFILE" ]]; then
        N_EXISTING=$(grep -c "^>" "$OUTFILE" 2>/dev/null || echo 0)
        if [[ "$N_EXISTING" -ge "$MAX" ]]; then
            echo "  [$FAMILY] Already have $N_EXISTING (target $MAX), skipping"
            TOTAL_DL=$((TOTAL_DL + N_EXISTING))
            continue
        fi
    fi

    echo ""
    echo "  [$FAMILY] Searching (max $MAX)..."

    # URL-encode the query using Python
    ENCODED_QUERY=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$QUERY'''))")
    
    # Search
    SEARCH_RESULT=$(curl -fsSL --max-time 30 \
        "${ESEARCH}?db=nucleotide&term=${ENCODED_QUERY}&retmax=${MAX}&usehistory=y" \
        2>/dev/null || echo "")

    if [[ -z "$SEARCH_RESULT" ]]; then
        echo "    WARNING: search failed for $FAMILY, skipping"
        continue
    fi

    # Extract WebEnv, QueryKey, Count
    WEBENV=$(echo "$SEARCH_RESULT" | grep -oP '<WebEnv>\K[^<]+' | head -1 || echo "")
    QKEY=$(echo "$SEARCH_RESULT" | grep -oP '<QueryKey>\K[^<]+' | head -1 || echo "")
    COUNT=$(echo "$SEARCH_RESULT" | grep -oP '<Count>\K[^<]+' | head -1 || echo "0")

    if [[ -z "$WEBENV" || -z "$QKEY" || "$COUNT" -eq 0 ]]; then
        echo "    No results (Count=$COUNT)"
        continue
    fi

    FETCH_N=$MAX
    if [[ "$COUNT" -lt "$MAX" ]]; then
        FETCH_N=$COUNT
    fi
    echo "    Found $COUNT, fetching $FETCH_N..."

    # Fetch in batches
    > "$TMP_DIR/${FAMILY}.fasta"
    RETSTART=0
    BATCH=200
    while [[ $RETSTART -lt $FETCH_N ]]; do
        CHUNK="$TMP_DIR/${FAMILY}_chunk_${RETSTART}.fasta"
        curl -fsSL --max-time 120 \
            "${EFETCH}?db=nucleotide&query_key=${QKEY}&WebEnv=${WEBENV}&rettype=fasta&retmode=text&retstart=${RETSTART}&retmax=${BATCH}" \
            -o "$CHUNK" 2>/dev/null || true
        
        if [[ -f "$CHUNK" ]] && [[ -s "$CHUNK" ]]; then
            cat "$CHUNK" >> "$TMP_DIR/${FAMILY}.fasta"
        fi
        rm -f "$CHUNK"
        RETSTART=$((RETSTART + BATCH))
        sleep 0.35  # NCBI rate limit
    done

    # Count and move
    N_SEQS=$(grep -c "^>" "$TMP_DIR/${FAMILY}.fasta" 2>/dev/null || echo 0)
    if [[ "$N_SEQS" -gt 0 ]]; then
        mv "$TMP_DIR/${FAMILY}.fasta" "$OUTFILE"
        echo "    [$FAMILY] Saved $N_SEQS sequences"
    else
        echo "    [$FAMILY] 0 sequences fetched"
    fi
    TOTAL_DL=$((TOTAL_DL + N_SEQS))
    echo "    Running total: $TOTAL_DL sequences"
done

log "Done! Total downloaded: $TOTAL_DL sequences across families"
echo ""
echo "Next: python experiments/exp1_datascale/merge_10k_corpus.py"
