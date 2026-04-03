# Evo-Locking

Spectral Deformation (SpecDef) weight locking applied to [Evo](https://github.com/evo-design/evo) (StripedHyena, 6.4B parameters).

Based on: *Locking Open Weight Models with Spectral Deformation* — Rosati et al. (ICML 2025 Workshop).

## Overview

SpecDef inflates the top-`k` singular values of weight matrices to make fine-tuning ill-conditioned. This raises the curvature of the loss landscape (Theorem 2.1), so an attacker who fine-tunes locked weights faces gradient instability and slow convergence.

**This repo applies SpecDef to all Hyena blocks in Evo**, targeting every `nn.Linear` layer (projections, output dense, and MLP) across all 32 blocks.

The pipeline runs in two stages:
1. **Lock** — SpecDef locking on retain data (bacterial/phage genomes)
2. **Finetune** — attack fine-tuning on a held-out distribution (eukaryotic viruses), run twice: once from the locked checkpoint, once from the pretrained baseline

---

## Repo Structure

```
evo-locking/
├── src/
│   └── utils.py                  # Dataclasses, data loading, SVD, loss, DDP helpers
├── scripts/
│   ├── lock.py                   # Stage 1: SpecDef locking
│   ├── finetune.py               # Stage 2: fine-tune attack
│   ├── eval_pretrained.py        # Evaluate pretrained model on attack data
│   ├── run.sh                    # Single-job dispatcher (1 GPU or torchrun)
│   ├── run_pipeline_v10.sh       # Full pipeline: lock → finetune locked + unlocked
│   └── test_pipeline.sh          # Config/import smoke tests (no GPU needed)
├── configs/
│   ├── lock_v10_full.yaml        # DEFINITIVE: 32 blocks, top_k=5, 10k steps (H100-80GB)
│   ├── ft_attack_v10_full.yaml   # DEFINITIVE: 32 blocks, 20k steps (H100-80GB)
│   ├── lock_v10_h100.yaml        # 24 blocks, 5k steps (H100-80GB)
│   ├── ft_attack_v10_h100.yaml   # 24 blocks, 5k steps (H100-80GB)
│   ├── lock_v10_a100.yaml        #  8 blocks, 5k steps (A100-40GB)
│   └── ft_attack_v10_a100.yaml   #  8 blocks, 5k steps (A100-40GB)
├── data/
│   ├── retain.fasta              # 2706 bacterial genomes + 481 phages (134.9 MB)
│   └── attack.fasta              # 910 eukaryotic virus genomes (38.7 MB)
├── results/                      # Output directory (gitignored)
├── locking.pdf                   # SpecDef paper
└── README.md
```

---

## Quick Start

### Run the full definitive experiment (H100 80GB)

```bash
sbatch -p h100 --gres=gpu:1 -t 72:00:00 scripts/run_pipeline_v10.sh full
```

This runs:
1. Lock — 32 blocks, `top_k=5`, 10 000 steps → `results/lock_v10_full/model_locked.pt`
2. Finetune locked — 32 blocks, 20 000 steps → `results/ft_attack_v10_full_locked/`
3. Finetune unlocked — 32 blocks, 20 000 steps → `results/ft_attack_v10_full_unlocked/`

A summary table comparing locked vs unlocked val loss is printed at the end.

### Other GPU tiers

```bash
# H100 80GB — 24 blocks (faster iteration)
sbatch -p h100         --gres=gpu:1 scripts/run_pipeline_v10.sh h100

# A100 40GB — 8 blocks (smoke test, fits ~31 GB)
sbatch -p gpu-a100     --gres=gpu:1 scripts/run_pipeline_v10.sh a100
sbatch -p gpu-a100-dev --gres=gpu:1 scripts/run_pipeline_v10.sh a100
```

### Skip re-locking (reuse existing checkpoint)

```bash
sbatch -p h100 --gres=gpu:1 scripts/run_pipeline_v10.sh full --skip-lock
```

### Run stages manually

```bash
# Single-GPU lock from any config
bash scripts/run.sh lock configs/lock_v10_full.yaml

# Single-GPU finetune (mode=both → runs locked then unlocked)
bash scripts/run.sh finetune configs/ft_attack_v10_full.yaml

# Multi-GPU (torchrun auto-detected when SLURM allocates >1 GPU)
sbatch -p h100 --gres=gpu:4 scripts/run_pipeline_v10.sh full
```

---

## GPU Tier Reference

| Tier | Config | Target blocks | top_k | Lock steps | FT steps | Peak memory |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| `full` | `lock_v10_full.yaml` | 32 | 5 | 10 000 | 20 000 | ~79 GB |
| `h100` | `lock_v10_h100.yaml` | 24 | 3 | 5 000 | 5 000 | ~65 GB |
| `a100` | `lock_v10_a100.yaml` | 8 | 3 | 5 000 | 5 000 | ~31 GB |

For the `full` tier: lock covers ~0.67 epochs of retain data; finetune covers ~2.15 epochs of attack data.

---

## Config Format (YAML)

All hyperparameters live in YAML files under `configs/`. Scripts accept `--config <path>`.

### Lock config (`LockConfig`)

```yaml
run_name: lock_v10_full        # results saved to results/{run_name}/
retain_data_path: data/retain.fasta
seed: 42
train_fraction: 0.9
min_seq_len: 512

lock_steps: 10000
lock_lr: 5e-5
alpha_start: 0.8              # linear schedule: utility loss weight early
alpha_end: 0.3                # pushes spectral inflation toward the end
top_k: 5                      # number of singular values inflated per matrix

batch_size: 8
seq_len: 1024
grad_accum_steps: 1
val_every: 100
val_batches: 8
max_grad_norm: 1.0

target_blocks: 32             # integer → blocks 0..(N-1); list → specific blocks
target_layer_patterns:        # substring patterns that identify locked weight matrices
  - .projections.weight
  - .out_filter_dense.weight
  - .mlp.l1.weight
  - .mlp.l2.weight
  - .mlp.l3.weight
```

### Finetune config (`FinetuneConfig`)

```yaml
mode: both                    # 'locked' | 'unlocked' | 'both' (runs both sequentially)
run_name: ft_attack_v10_full  # suffixed: _locked / _unlocked per mode
locked_ckpt: results/lock_v10_full/model_locked.pt  # required when mode includes 'locked'

data_path: data/attack.fasta
seed: 42
train_fraction: 0.9
min_seq_len: 1024

train_steps: 20000
lr: 1e-5
batch_size: 1
seq_len: 1024
grad_accum_steps: 4
val_every: 200
eval_batches: 16
max_grad_norm: 1.0
optimizer_name: adamw

target_blocks: 32
save_checkpoint: true
use_gradient_checkpointing: true
```

Results are written to:
- `results/{run_name}_locked/`  — `metrics.csv`, `model_finetuned.pt`
- `results/{run_name}_unlocked/` — same structure

---

## Datasets

### Retain (`data/retain.fasta`)

- 2706 bacterial genome windows (E. coli, Klebsiella, Pseudomonas, etc.) + 481 phage sequences
- 134.9 MB, ~119 400 windows of 1024 bp at 90/10 train/val split
- Similar distribution to Evo's OpenGenome pretraining corpus
- Purpose: preserve Evo's general genomic utility during locking

### Attack (`data/attack.fasta`)

- 910 eukaryotic virus genomes: Herpesviridae, Adenoviridae, Papillomaviridae, Coronaviridae, Poxviridae, Retroviridae, Flaviviridae, Filoviridae, etc.
- 38.7 MB, ~37 100 windows of 1024 bp at 90/10 train/val split
- **Not in Evo's pretraining** — Evo was pretrained on prokaryotic/viral (phage) sequences via IMG/VR; eukaryotic-infecting viruses were excluded from OpenGenome
- Purpose: out-of-distribution fine-tune target — tests whether locking prevents memorization of a genuinely novel sequence distribution

---

## Architecture Notes

Evo-1-8k-base is a StripedHyena model:
- 32 blocks total: 29 Hyena (SSM) + 3 Attention (at positions 8, 16, 24)
- `hidden_size=4096`, `vocab_size=512`, `max_seq_len=8192`
- Each Hyena block: `projections` → SSM filter → `out_filter_dense` → MLP (SwiGLU: `l1`, `l2`, `l3`)
- **Locked**: all 5 `nn.Linear` families per block → 160 matrices when targeting all 32 blocks
- **Not locked**: SSM filter parameters (poles, residues, `short_filter`) — ~0.03% of block params, sandwiched between locked layers

---

## DDP (Multi-GPU)

`run.sh` auto-detects the GPU count from `SLURM_GPUS_ON_NODE` and calls `torchrun --standalone --nproc_per_node=N` when N > 1. Both `lock.py` and `finetune.py` support DDP via `setup_ddp()` / `wrap_ddp()` / `cleanup_ddp()` in `src/utils.py`:

- Device is assigned by `LOCAL_RANK`
- Data is split deterministically across ranks (no duplicates)
- Per-rank seed = `cfg.seed + rank`
- `model.no_sync()` is used during gradient accumulation to avoid redundant all-reduces
- Only rank-0 writes logs, metrics CSV, and checkpoints
- Checkpoints save `raw_model.state_dict()` (unwrapped from DDP)

---

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA + NCCL (bf16 support required)
- `evo` package (`evo-design/evo`)
- `PyYAML >= 6.0`
- `matplotlib` (optional, for plots)
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for H100 runs near the memory limit
