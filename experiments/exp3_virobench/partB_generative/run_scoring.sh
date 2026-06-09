#!/bin/bash
# Exp 3B Step 2 — Score generated sequences using geNomad, Pyrodigal, and BLAST.
#
# For each generated sequence:
#   1. geNomad: viral classification score (is it recognizably viral?)
#   2. Pyrodigal: coding density (fraction in plausible ORFs)
#   3. BLAST: identity to training corpus (memorization check)
#
# Usage:
#   bash experiments/exp3_virobench/partB_generative/run_scoring.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
GEN_DIR="$SCRIPT_DIR/generated"
SCORE_DIR="$SCRIPT_DIR/scored"
BLAST_DB="$SCRIPT_DIR/blastdb/train_corpus"

mkdir -p "$SCORE_DIR"

echo "=== Exp 3B — Sequence Scoring ==="
echo ""

for MODEL in unlocked_ft m_locked; do
    echo "--- Scoring $MODEL ---"
    GEN_SUBDIR="$GEN_DIR/$MODEL"
    SCORE_CSV="$SCORE_DIR/${MODEL}_scores.csv"
    echo "seq_id,genomad_score,genomad_label,coding_density,num_orfs,max_orf_len,blast_identity,blast_evalue,blast_label" > "$SCORE_CSV"

    for fasta_file in "$GEN_SUBDIR"/*.fasta; do
        [ -f "$fasta_file" ] || continue
        seq_id=$(basename "$fasta_file" .fasta)

        # ── 1. geNomad ──────────────────────────────────
        genomad_score="NA"
        genomad_label="NA"
        if command -v genomad &>/dev/null; then
            TMP_GENOMAD=$(mktemp -d)
            genomad end-to-end --cleanup "$fasta_file" "$TMP_GENOMAD/genomad_out" /tmp/genomad_db 2>/dev/null || true
            if [ -f "$TMP_GENOMAD/genomad_out/${seq_id}_summary/${seq_id}_virus_summary.tsv" ]; then
                genomad_score=$(tail -n +2 "$TMP_GENOMAD/genomad_out/${seq_id}_summary/${seq_id}_virus_summary.tsv" 2>/dev/null | cut -f5 | head -1 || echo "NA")
                genomad_label="viral"  # placeholder
            fi
            rm -rf "$TMP_GENOMAD"
        else
            echo "  [WARN] geNomad not found — skipping viral scoring"
        fi

        # ── 2. Pyrodigal ORF calling ────────────────────
        coding_density="NA"; num_orfs="NA"; max_orf_len="NA"
        if python3 -c "import pyrodigal" 2>/dev/null; then
            TMP_ORFS=$(mktemp)
            python3 -c "
from Bio import SeqIO
import pyrodigal
rec = next(SeqIO.parse('$fasta_file', 'fasta'))
orf_finder = pyrodigal.GeneFinder(meta=False)
genes = orf_finder.find_genes(str(rec.seq))
total_orf = sum(len(g) for g in genes)
orfs = [(len(g), str(g)) for g in genes]
if len(rec.seq) > 0:
    cd = total_orf / len(rec.seq)
else:
    cd = 0
print(f'{cd:.4f} {len(orfs)}', end='')
if orfs:
    print(f' {max(o[0] for o in orfs)}', end='')
else:
    print(' 0', end='')
" > "$TMP_ORFS" 2>/dev/null
            coding_density=$(awk '{print $1}' "$TMP_ORFS" 2>/dev/null || echo "NA")
            num_orfs=$(awk '{print $2}' "$TMP_ORFS" 2>/dev/null || echo "NA")
            max_orf_len=$(awk '{print $3}' "$TMP_ORFS" 2>/dev/null || echo "NA")
            rm -f "$TMP_ORFS"
        else
            echo "  [WARN] Pyrodigal not found — skipping ORF calling"
        fi

        # ── 3. BLAST vs training corpus ─────────────────
        blast_identity="NA"; blast_evalue="NA"; blast_label="NA"
        if command -v blastn &>/dev/null && [ -f "$BLAST_DB.nhr" ]; then
            TMP_BLAST=$(mktemp)
            blastn -db "$BLAST_DB" -query "$fasta_file" -outfmt "6 qseqid sseqid pident evalue" \
                -max_target_seqs 1 -num_threads 4 > "$TMP_BLAST" 2>/dev/null || true
            if [ -s "$TMP_BLAST" ]; then
                blast_identity=$(awk '{print $3}' "$TMP_BLAST" | head -1)
                blast_evalue=$(awk '{print $4}' "$TMP_BLAST" | head -1)
                # Label: >95% = memorization, 70-95% = related, <70% = novel
                if [ -n "$blast_identity" ]; then
                    if (( $(echo "$blast_identity > 95" | bc -l 2>/dev/null || echo 0) )); then
                        blast_label="memorization"
                    elif (( $(echo "$blast_identity > 70" | bc -l 2>/dev/null || echo 0) )); then
                        blast_label="related"
                    else
                        blast_label="novel"
                    fi
                fi
            fi
            rm -f "$TMP_BLAST"
        else
            echo "  [WARN] BLAST not found or DB missing — skipping identity check"
        fi

        echo "$seq_id,$genomad_score,$genomad_label,$coding_density,$num_orfs,$max_orf_len,$blast_identity,$blast_evalue,$blast_label" >> "$SCORE_CSV"
    done

    n=$(tail -n +2 "$SCORE_CSV" | wc -l)
    echo "  Scored $n sequences → $SCORE_CSV"
done

echo ""
echo "=== Scoring complete ==="
echo "Next: python3 $SCRIPT_DIR/compare_distributions.py"
