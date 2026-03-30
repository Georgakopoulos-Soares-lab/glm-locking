#!/usr/bin/env bash
# =============================================================================
# download_attack.sh  –  Download human-host virus genomes for the finetune attack
#
# Fetches 10 human-infecting virus reference genomes from NCBI GenBank.
# These viruses were excluded from Evo-1's OpenGenome pretraining (which
# covered only prokaryotes and prokaryotic phages), making them genuinely
# out-of-distribution for the base model.
#
# Output: data/attack.fasta  (~2.8 MB, 15 sequences: 9 viruses + 6 influenza segments)
#
# Requirements: curl  (standard on any Linux/HPC node)
# No credentials needed — NCBI efetch is public.
#
# Usage:
#   bash data/download_scripts/download_attack.sh
#
# Virus list:
#   1.  SARS-CoV-2          NC_045512.2   (29.9 kb, +ssRNA betacoronavirus)
#   2.  HIV-1 HXB2          K03455.1      ( 9.7 kb, retrovirus)
#   3.  Herpes Simplex 1    NC_001806.2   (152 kb,  dsDNA alphaherpesvirus)
#   4.  Epstein-Barr Virus  NC_007605.1   (172 kb,  dsDNA gammaherpesvirus)
#   5.  CMV (HHV-5)         NC_006273.2   (236 kb,  dsDNA betaherpesvirus)
#   6.  Influenza A H1N1    NC_026433.1 – NC_026438.1  (6 segments, ~12 kb)
#   7.  Ebola (Zaire)       NC_002549.1   ( 19 kb,  -ssRNA filovirus)
#   8.  Dengue Virus 1      NC_001477.1   ( 10.7 kb, +ssRNA flavivirus)
#   9.  Human Adenovirus 5  AC_000008.1   ( 35.9 kb, dsDNA adenovirus)
#  10.  Vaccinia Virus       NC_006998.1   (195 kb,  dsDNA poxvirus)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$PROJECT_DIR/data/attack.fasta"

log()  { echo ""; echo ">>> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# Check for curl
command -v curl >/dev/null 2>&1 || die "curl not found. Install with: apt install curl / brew install curl"

EFETCH="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

fetch_fasta() {
    local acc="$1"
    local label="$2"
    echo "  Fetching $acc ($label)..."
    # rettype=fasta retmode=text — plain FASTA from nucleotide db
    curl -fsSL \
        "$EFETCH?db=nucleotide&id=${acc}&rettype=fasta&retmode=text" \
        --retry 3 --retry-delay 2 2>/dev/null
    # Small sleep to be polite to NCBI (max 3 req/s without API key)
    sleep 0.4
}

# ---------------------------------------------------------------------------
# Sequences to download
# ---------------------------------------------------------------------------
declare -a ACCESSIONS=(
    # acc             label
    "NC_045512.2"     "SARS-CoV-2"
    "K03455.1"        "HIV-1_HXB2"
    "NC_001806.2"     "Herpes_Simplex_Virus_1"
    "NC_007605.1"     "Epstein-Barr_Virus"
    "NC_006273.2"     "Human_Cytomegalovirus"
    # Influenza A H1N1 pdm09 (A/California/07/2009) — 6 of 8 segments (RefSeq NC_026433-NC_026438)
    "NC_026438.1"     "Influenza_A_H1N1_seg1_PB2"
    "NC_026435.1"     "Influenza_A_H1N1_seg2_PB1"
    "NC_026437.1"     "Influenza_A_H1N1_seg3_PA"
    "NC_026433.1"     "Influenza_A_H1N1_seg4_HA"
    "NC_026436.1"     "Influenza_A_H1N1_seg5_NP"
    "NC_026434.1"     "Influenza_A_H1N1_seg6_NA"
    "NC_002549.1"     "Ebola_Zaire"
    "NC_001477.1"     "Dengue_Virus_1"
    "AC_000008.1"     "Human_Adenovirus_5"
    "NC_006998.1"     "Vaccinia_Virus"
)

# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------
if [[ -f "$OUT" ]]; then
    N=$(grep -c "^>" "$OUT" 2>/dev/null || echo 0)
    log "attack.fasta already exists ($N sequences). Delete to re-download."
    exit 0
fi

log "Downloading human-host virus genomes from NCBI..."
mkdir -p "$PROJECT_DIR/data"

TMP="$(mktemp /tmp/attack_fasta.XXXXXX)"
trap 'rm -f "$TMP"' EXIT

# Iterate in pairs: accession then label
i=0
while [[ $i -lt ${#ACCESSIONS[@]} ]]; do
    acc="${ACCESSIONS[$i]}"
    label="${ACCESSIONS[$((i+1))]}"
    fetch_fasta "$acc" "$label" >> "$TMP"
    i=$((i+2))
done

# Sanity check
N=$(grep -c "^>" "$TMP")
echo ""
echo "Downloaded $N FASTA sequences."
[[ $N -ge 17 ]] || { echo "WARNING: expected >=17 sequences, got $N — check network/NCBI availability"; }

# Clean: strip blank lines between records, ensure Unix line endings
awk '/^>/{if(seq) print seq; print; seq=""} !/^>/{seq=seq $0} END{if(seq) print seq}' "$TMP" > "$OUT"

log "Wrote $OUT"
echo ""
echo "Sequence breakdown:"
grep "^>" "$OUT" | sed 's/^/  /'
echo ""

# Summary of total bases
BASES=$(grep -v "^>" "$OUT" | tr -d '\n' | wc -c)
echo "Total bases: $(printf "%'d" $BASES)"
echo ""
echo "Done. Run python scripts/finetune.py to start the attack."
