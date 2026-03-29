# Evo-Locking

Spectral Deformation (SpecDef) weight locking applied to [Evo](https://github.com/evo-design/evo) (StripedHyena, 6.4B parameters).

Based on: *Locking Open Weight Models with Spectral Deformation* — Rosati et al. (ICML 2025 Workshop).

## Overview

SpecDef inflates the top singular values of weight matrices to make fine-tuning ill-conditioned. This raises the curvature of the loss landscape (via Theorem 2.1), so an attacker who tries to fine-tune the locked weights faces exploding gradients.

**This repo applies SpecDef to Evo's Hyena blocks** — targeting all `nn.Linear` layers (projections, output dense, and MLP) across all 32 blocks.

## Repo Structure

```
evo-locking/
├── src/
│   └── utils.py                    # Shared utilities, configs, data loading, SVD monitoring
├── scripts/
│   ├── lock.py                     # Step 1: SpecDef locking (inflate singular values)
│   ├── run_lock_slurm.sh           # SLURM launcher for lock.py (1 or N GPUs)
│   ├── finetune.py                 # Step 2: Fine-tune attack (locked or unlocked init)
│   └── eval_pretrained.py         # Step 0: Pretrained baseline evaluation
├── data/
│   ├── download_scripts/           # Reproducible pipeline to build retain.fasta
│   │   ├── prepare_retain.sh      # Full pipeline: all downloads + processing
│   │   ├── retain.py              # Python processing (4 subcommands)
│   │   ├── .env                   # JGI credentials (gitignored)
│   │   └── .env.example           # Credential template (copy to .env)
│   └── retain.fasta               # Built by download pipeline (gitignored)
├── old/                            # Archived v7 scripts
├── results/                        # Output directory (gitignored)
├── locking.pdf                     # SpecDef paper
└── README.md
```

## Requirements

### Python environment

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/evo-design/evo.git
pip install matplotlib tqdm
```

Python 3.10+ required. CUDA with bfloat16 support recommended (A100/H100).

### Data download (CPU-only, no pip needed)

Standard tools available on any Linux/HPC login node:
```
curl   wget   python3   awk   zcat   xxd   unzip
```

JGI account required for IMG/VR data — register free at https://contacts.jgi.doe.gov/registration/new

## Data Setup

### Build retain.fasta

The retain set is built from GTDB r220 (bacteria/archaea) + IMG/VR v4 (prokaryotic phages) — ~6,600 sequences, ~2 GB.

**1. Set JGI credentials:**

```bash
cp data/download_scripts/.env.example data/download_scripts/.env
# Edit .env: fill in JGI_USER and JGI_PASS
```

**2. (Optional) set scratch directory** (~50 GB free needed):

```bash
export SCRATCH_DIR=/path/to/scratch   # default: /scratch/10906/arisk/evo_locking_data
```

**3. Run the full pipeline** (1–2 h, mostly download time — use screen/tmux):

```bash
bash data/download_scripts/prepare_retain.sh
```

All steps are idempotent — re-running skips completed steps automatically. The final `data/retain.fasta` is a symlink to `$SCRATCH_DIR/retain.fasta`.

Individual Python steps can also be run standalone:

```bash
python3 data/download_scripts/retain.py --help
python3 data/download_scripts/retain.py sample-gtdb
python3 data/download_scripts/retain.py extract-contigs
python3 data/download_scripts/retain.py sample-imgvr
python3 data/download_scripts/retain.py build
```

For full details see [data/download_scripts/README.md](data/download_scripts/README.md).

## Experimental Pipeline

All scripts below run on a **GPU node** with the Evo model loaded.

### Step 0: Pretrained baseline

```bash
python scripts/eval_pretrained.py
```

Measures pretrained Evo loss/perplexity/accuracy before any locking.

### Step 1: Lock the model (SpecDef)

**Single GPU:**
```bash
python scripts/lock.py
```

**Multi-GPU (auto-detected):**
```bash
python scripts/lock.py          # detects all GPUs automatically, re-launches via torchrun
torchrun --nproc_per_node=3 scripts/lock.py   # explicit
```

**Via SLURM:**
```bash
sbatch --gres=gpu:1 scripts/run_lock_slurm.sh   # single GPU
sbatch --gres=gpu:3 scripts/run_lock_slurm.sh   # 3-GPU DDP
```

The locking procedure:

- Loads `data/retain.fasta` (bacteria/archaea/phage, not task-specific)
- Targets all `nn.Linear` weights in all 32 Hyena blocks (160 matrices):
  - `projections.weight` — input projections
  - `out_filter_dense.weight` — output dense
  - `mlp.l1.weight`, `mlp.l2.weight`, `mlp.l3.weight` — SwiGLU MLP
- Loss: $\mathcal{L}_\text{lock} = \alpha \cdot \mathcal{L}_\text{retain} - (1-\alpha) \cdot \overline{\sigma_1}$
- Alpha anneals linearly 0.8 → 0.3 over 500 steps (preserve utility early, maximize inflation late)
- Spectral term uses **randomized SVD** (`torch.svd_lowrank`) — stays on GPU, 50x faster than full SVD, differentiable
- DDP: spectral backward runs inside `no_sync()` (weights are identical across ranks; no all_reduce needed), retain backward syncs normally
- Saves checkpoint to `results/lock_v8_all_linear/model_locked.pt`

Current config (`scripts/lock.py`):

| Parameter | Value |
|-----------|-------|
| `lock_steps` | 500 |
| `lock_lr` | 5e-5 |
| `alpha_start / alpha_end` | 0.8 → 0.3 |
| `top_k` | 1 |
| `batch_size` | 4 (per GPU) |
| `seq_len` | 1024 |
| `grad_accum_steps` | 1 |
| `target_blocks` | all 32 |

### Step 2: Fine-tune attack

Run twice to compare locked vs unlocked:

```bash
# On locked model (set CONFIG.locked_ckpt = "results/lock_v8_all_linear/model_locked.pt")
python scripts/finetune.py

# On unlocked model (set CONFIG.locked_ckpt = None, different run_name)
python scripts/finetune.py
```

If locking is effective, the locked run shows higher val loss and lower next-token accuracy than the unlocked control.

## Evo Architecture Notes

Evo-1-8k-base is a StripedHyena model:
- 32 blocks total: 29 Hyena (SSM) + 3 Attention (at layers 8, 16, 24)
- `hidden_size=4096`, `vocab_size=512`, `max_seq_len=8192`
- Each Hyena block: projections → SSM filter → out_filter_dense → MLP (SwiGLU)
- SSM filter parameters (poles, residues, short_filter) are **not** locked — they are ~0.03% of block params

## Key Changes from v7

| Aspect | v7 | Current |
|--------|----|---------| 
| Lock targets | `projections.weight` only (8 matrices) | All `nn.Linear` in all 32 blocks (160 matrices) |
| Spectral aggregation | `.sum()` | `.mean()` (scale-stable) |
| Lock steps | 50 | 500 |
| Seq len (lock) | 64 | 1024 |
| Seq len (fine-tune) | 128 | 1024 |
| SVD method | Full `svdvals` on GPU | Randomized `svd_lowrank` on GPU (no OOM, 50× faster) |
| Gradient accumulation | None | Supported (default 1) |
| Alpha schedule | Fixed 0.5 | Linear 0.8 → 0.3 |
| SVD monitoring | None | Before/after per-matrix logging |
| Multi-GPU | None | DDP via torchrun, auto-detected |
| Retain dataset | None | GTDB r220 + IMG/VR v4 (6,600 seqs, ~2 GB) |
| Data split | Same data for lock + attack | Separate retain/attack datasets |

