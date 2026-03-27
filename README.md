# Evo-Locking

Spectral Deformation (SpecDef) weight locking applied to [Evo](https://github.com/evo-design/evo) (StripedHyena, 6.4B parameters).

Based on: *Locking Open Weight Models with Spectral Deformation* — Rosati et al. (ICML 2025 Workshop).

## Overview

SpecDef inflates the top singular values of weight matrices to make fine-tuning ill-conditioned. This raises the curvature of the loss landscape (via Theorem 2.1), so an attacker who tries to fine-tune the locked weights faces exploding gradients.

**This repo applies SpecDef to Evo's Hyena blocks** — targeting all `nn.Linear` layers (projections, output dense, and MLP) in blocks 0–7.

## Repo Structure

```
evo-locking/
├── src/
│   └── utils.py                          # Shared utilities, configs, data loading, SVD monitoring
├── scripts/
│   ├── lock.py                           # Step 1: SpecDef locking (inflate singular values)
│   ├── finetune.py                       # Step 2: Fine-tune attack (locked or unlocked init)
│   └── eval_pretrained.py               # Step 0: Pretrained baseline evaluation
├── data/
│   ├── download_scripts/                 # Reproducible pipeline to build retain.fasta
│   │   ├── 00_setup_env.sh              # Build seqtk from source (no conda needed)
│   │   ├── 01_download_gtdb_metadata.sh # GTDB r220 taxonomy TSVs (~21 MB)
│   │   ├── 02_sample_gtdb_accessions.py # Sample 5,500 genomes proportionally
│   │   ├── 03_download_gtdb_genomes.sh  # Download only the sampled genomes (~15-25 GB)
│   │   ├── 04_extract_gtdb_contigs.py   # 1 contig ≥10kb per genome → FASTA
│   │   ├── 05_download_imgvr_metadata.sh# IMG/VR v4 sequence info TSV (~200 MB)
│   │   ├── 06_sample_imgvr.py           # Filter/sample ~3,000 prokaryotic phage IDs
│   │   ├── 07_stream_imgvr_sequences.sh # Stream-extract sequences (no 30 GB temp file)
│   │   ├── 08_build_retain.py           # Combine GTDB + IMG/VR → retain.fasta
│   │   ├── run_all.sh                   # Master script: runs all steps end-to-end
│   │   └── .env.example                 # Credential template (copy to .env)
│   ├── retain.fasta                     # Built by download pipeline (gitignored)
│   └── attack.fasta                     # Task-specific fine-tune data (bring your own)
├── old/                                  # Archived v7 scripts
├── results/                              # Output directory (gitignored)
├── locking.pdf                           # SpecDef paper
└── README.md
```

## Requirements

### Python environment (GPU node)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/evo-design/evo.git
pip install matplotlib tqdm
```

Python 3.10+ required. CUDA with bfloat16 support recommended (A100/H100).

### Data download node (CPU-only)

No Python packages needed beyond the standard library. The pipeline builds `seqtk` from source automatically via `00_setup_env.sh`.

You need:
- `curl`, `git`, `make`, `gcc` (standard on HPC login nodes)
- JGI account for IMG/VR data — register free at https://contacts.jgi.doe.gov/registration/new

## Data Setup

You need two FASTA files in `data/` before running experiments:

| File | Purpose | Source |
|------|---------|--------|
| `data/retain.fasta` | General-purpose DNA for locking (preserve utility) | Built by download pipeline below |
| `data/attack.fasta` | Task-specific data the attacker fine-tunes on | Bring your own (e.g., viral genomes) |

The retain set must be a **different** distribution from the attack data. This pipeline builds it from GTDB r220 (bacteria/archaea) + IMG/VR v4 (prokaryotic phages).

### Build retain.fasta

**1. Set up JGI credentials** (needed for IMG/VR download):

```bash
cp data/download_scripts/.env.example data/download_scripts/.env
# Edit .env and fill in your JGI email and password
```

**2. Set scratch directory** (where large files are stored):

```bash
export SCRATCH_DIR=/path/to/your/scratch   # default: /scratch/10906/arisk/evo_locking_data
```

**3. Run the full pipeline** (~15–25 GB total download, mostly GTDB genomes):

```bash
bash data/download_scripts/run_all.sh
```

Or run steps individually:

```bash
# Prerequisites: build seqtk from source
bash data/download_scripts/00_setup_env.sh

# GTDB r220: metadata → sample IDs → download only selected genomes → extract contigs
bash data/download_scripts/01_download_gtdb_metadata.sh  # ~21 MB
python3 data/download_scripts/02_sample_gtdb_accessions.py
bash data/download_scripts/03_download_gtdb_genomes.sh   # ~15-25 GB (5,500 genomes)
python3 data/download_scripts/04_extract_gtdb_contigs.py

# IMG/VR v4: metadata → sample IDs → stream-extract sequences (30 GB never stored)
bash data/download_scripts/05_download_imgvr_metadata.sh # ~200 MB TSV
python3 data/download_scripts/06_sample_imgvr.py
bash data/download_scripts/07_stream_imgvr_sequences.sh  # ~50-150 MB output

# Combine into retain.fasta
python3 data/download_scripts/08_build_retain.py
```

The final `data/retain.fasta` symlink points to `$SCRATCH_DIR/retain.fasta` (~8,500 sequences, mixed bacteria/archaea/phage).

## Experimental Pipeline

All scripts below must run on a **GPU node** with the Evo model loaded.

### Step 0: Pretrained baseline

```bash
python scripts/eval_pretrained.py
```

Measures pretrained Evo loss/perplexity/accuracy on the attack dataset before any locking.

### Step 1: Lock the model (SpecDef)

```bash
python scripts/lock.py
```

- Loads `data/retain.fasta` (general-purpose, not the attack task)
- Targets all `nn.Linear` weights in Hyena blocks 0–7 (40 matrices):
  - `projections.weight` (input projections, 12288 × 4096)
  - `out_filter_dense.weight` (output dense, 4096 × 4096)
  - `mlp.l1.weight`, `mlp.l2.weight`, `mlp.l3.weight` (SwiGLU MLP)
- Optimizes: $\mathcal{L}_\text{lock} = \alpha \cdot \mathcal{L}_\text{retain} - (1-\alpha) \cdot \text{spectral\_term}$
- Alpha schedules linearly 0.8 → 0.3 (preserve utility early, maximize inflation late)
- Saves locked checkpoint to `results/lock_v8_all_linear/model_locked.pt`

### Step 2: Fine-tune attack (locked vs unlocked)

Run **twice** to compare:

```bash
# Attack on locked model (edit CONFIG.locked_ckpt first)
python scripts/finetune.py

# Attack on unlocked model (set CONFIG.locked_ckpt = None, CONFIG.run_name = "ft_attack_unlocked_v8")
python scripts/finetune.py
```

If locking is effective, the locked run shows higher val loss and lower next-token accuracy than the unlocked control.

## Evo Architecture Notes

Evo-1-8k-base is a StripedHyena model:
- 32 blocks total: 29 Hyena (SSM) + 3 Attention (at layers 8, 16, 24)
- hidden_size=4096, vocab_size=512, max_seq_len=8192
- Each Hyena block: projections → SSM filter → out_filter_dense → MLP (SwiGLU)
- SSM filter parameters (poles, residues, short_filter) are **not** locked — they are ~0.03% of block params and sit between locked linear layers

## Key Changes from v7

| Aspect | v7 | v8 (current) |
|--------|-----|--------------|
| Lock targets | `projections.weight` only (8 matrices) | All `nn.Linear` in blocks (40 matrices) |
| Spectral aggregation | `.sum()` | `.mean()` (scale-stable) |
| Lock steps | 50 | 500 |
| Seq len (lock) | 64 | 512 |
| Seq len (fine-tune) | 128 | 1024 |
| Gradient accumulation | None | 4 steps |
| Alpha schedule | Fixed 0.5 | Linear 0.8 → 0.3 |
| SVD monitoring | None | Before/after logging |
| Data split | Same data for lock + attack | Separate retain/attack datasets |
