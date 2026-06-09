#!/bin/bash
# Exp 3B Part B Step 1 — Generate sequences from Unlocked-FT and M models.
#
# Uses Evo-1-8k-base's generate() method with matched sampling settings.
# Generates N sequences per model from viral-family-specific prompts.
#
# Usage:
#   bash experiments/exp3_virobench/partB_generative/run_generation.sh [GPU_ID]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
GPU="${1:-0}"
PYTHON=/home/nvidia/miniconda3/envs/evo/bin/python3

GEN_DIR="$SCRIPT_DIR/generated"
mkdir -p "$GEN_DIR/unlocked_ft" "$GEN_DIR/m_locked"

echo "=== Exp 3B — Sequence Generation ==="
echo "GPU: $GPU"
echo ""

# ── Generation parameters (MUST be identical across models) ──
PROMPT_LEN=256
GEN_LEN=768
TEMP=0.8
TOP_P=0.95
N_SEQS=100
SEED=42

# ── Prompt sequences (one per viral family class) ──
# These are short natural viral sequences from the training set.
# Extract them from the existing train.fasta
PROMPT_DIR="$SCRIPT_DIR/prompts"
mkdir -p "$PROMPT_DIR"

# Use the first 256nt of the first 3 training genomes as prompts
echo "Extracting prompt sequences from training data..."
$PYTHON -c "
from Bio import SeqIO
import os, random
random.seed($SEED)
recs = list(SeqIO.parse('$REPO_ROOT/data/attack_train.fasta', 'fasta'))
# Pick 3 diverse prompts
prompts = random.sample([r for r in recs if len(r.seq) >= $PROMPT_LEN], 3)
for i, r in enumerate(prompts):
    prompt_seq = str(r.seq[:$PROMPT_LEN])
    with open(f'$PROMPT_DIR/prompt_{i}.fasta', 'w') as f:
        f.write(f'>{r.id}\n{prompt_seq}\n')
    print(f'  Prompt {i}: {r.id} ({len(prompt_seq)} nt)')
" 2>/dev/null || echo "  BioPython not available — create prompts manually"

echo ""
echo "Generation settings:"
echo "  prompt_length:      $PROMPT_LEN"
echo "  generation_length:  $GEN_LEN"
echo "  temperature:        $TEMP"
echo "  top_p:              $TOP_P"
echo "  num_sequences:      $N_SEQS"
echo "  random_seed:        $SEED"
echo ""

# ── Generate from Unlocked-FT ───────────────────────────────
echo "=== Generating from Unlocked-FT ==="
CUDA_VISIBLE_DEVICES="$GPU" $PYTHON -c "
import torch, os, sys
sys.path.insert(0, '$REPO_ROOT')
from src.utils import load_evo_model, maybe_load_locked_checkpoint

model, tok = load_evo_model('evo-1-8k-base', 'cuda')
maybe_load_locked_checkpoint(model, '$REPO_ROOT/results/ft_unlocked_25k_v2_unlocked/model_best.pt')
model.eval()

prompt_dir = '$PROMPT_DIR'
out_dir = '$GEN_DIR/unlocked_ft'
os.makedirs(out_dir, exist_ok=True)

import glob
for prompt_file in sorted(glob.glob(f'{prompt_dir}/prompt_*.fasta')):
    with open(prompt_file) as f:
        lines = f.readlines()
    prompt_seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
    prompt_name = os.path.basename(prompt_file).replace('.fasta', '')

    input_ids = tok(prompt_seq, return_tensors='pt').input_ids.cuda()
    for i in range($N_SEQS):
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=$GEN_LEN,
                temperature=$TEMP,
                top_p=$TOP_P,
                do_sample=True,
                pad_token_id=tok.pad_token_id or 0,
            )
        gen_seq = tok.decode(out[0], skip_special_tokens=True)
        out_file = f'{out_dir}/{prompt_name}_gen_{i:04d}.fasta'
        with open(out_file, 'w') as f:
            f.write(f'>generated_{prompt_name}_{i:04d}\n{gen_seq}\n')
    print(f'  {prompt_name}: {($N_SEQS)} sequences generated')
print('  Unlocked-FT: DONE')
" 2>&1 | tail -5

echo ""

# ── Generate from M (α=3×10⁵ locked) ─────────────────────────
echo "=== Generating from M (α=3×10⁵) ==="
CUDA_VISIBLE_DEVICES="$GPU" $PYTHON -c "
import torch, os, sys
sys.path.insert(0, '$REPO_ROOT')
from src.utils import load_evo_model, maybe_load_locked_checkpoint

model, tok = load_evo_model('evo-1-8k-base', 'cuda')
maybe_load_locked_checkpoint(model, '$REPO_ROOT/results/ft_locked_a300k_lr1e5_25k_locked/model_best.pt')
model.eval()

prompt_dir = '$PROMPT_DIR'
out_dir = '$GEN_DIR/m_locked'
os.makedirs(out_dir, exist_ok=True)

import glob
for prompt_file in sorted(glob.glob(f'{prompt_dir}/prompt_*.fasta')):
    with open(prompt_file) as f:
        lines = f.readlines()
    prompt_seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
    prompt_name = os.path.basename(prompt_file).replace('.fasta', '')

    input_ids = tok(prompt_seq, return_tensors='pt').input_ids.cuda()
    for i in range($N_SEQS):
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=$GEN_LEN,
                temperature=$TEMP,
                top_p=$TOP_P,
                do_sample=True,
                pad_token_id=tok.pad_token_id or 0,
            )
        gen_seq = tok.decode(out[0], skip_special_tokens=True)
        out_file = f'{out_dir}/{prompt_name}_gen_{i:04d}.fasta'
        with open(out_file, 'w') as f:
            f.write(f'>generated_{prompt_name}_{i:04d}\n{gen_seq}\n')
    print(f'  {prompt_name}: {($N_SEQS)} sequences generated')
print('  M (α=3×10⁵): DONE')
" 2>&1 | tail -5

echo ""
echo "=== Generation complete ==="
echo "Output: $GEN_DIR/unlocked_ft/ and $GEN_DIR/m_locked/"
echo "Total sequences: $(ls $GEN_DIR/unlocked_ft/*.fasta 2>/dev/null | wc -l) + $(ls $GEN_DIR/m_locked/*.fasta 2>/dev/null | wc -l)"
echo ""
echo "Next: bash $SCRIPT_DIR/run_scoring.sh"
