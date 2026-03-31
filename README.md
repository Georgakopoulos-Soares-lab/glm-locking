# Evo-Locking

Spectral Deformation (SpecDef) weight locking applied to [Evo](https://github.com/evo-design/evo) (StripedHyena, 6.4B parameters).

Based on: *Locking Open Weight Models with Spectral Deformation* — Rosati et al. (ICML 2025 Workshop).

## Overview

SpecDef inflates the top singular values of weight matrices to make fine-tuning ill-conditioned. This raises the curvature of the loss landscape (via Theorem 2.1), so an attacker who tries to fine-tune the locked weights faces exploding gradients.

**This repo applies SpecDef to Evo's Hyena blocks** — targeting all `nn.Linear` layers (projections, output dense, and MLP) across all 32 blocks.

## Repo Structure

```
evo-locking/
├── configs/                         # YAML experiment configs (one file per run)
│   ├── lock_v8_all_linear.yaml      # Lock v8: 1000 steps, top_k=5  [DONE]
│   ├── lock_topk5_5000steps.yaml    # Lock v9: 5000 steps, top_k=5
│   ├── ft_locked_v8_1000lock_topk5_20ep.yaml    # Finetune on locked v8 checkpoint
│   ├── ft_unlocked_v8_1000lock_topk5_20ep.yaml  # Finetune unlocked baseline (v8 comparison)
│   ├── ft_locked_topk5_5000lock_20ep.yaml        # Finetune on locked v9 checkpoint
│   └── ft_unlocked_topk5_5000lock_20ep.yaml      # Finetune unlocked baseline (v9 comparison)
├── scripts/
│   ├── lock.py                      # SpecDef locking (inflate singular values)
│   ├── finetune.py                  # Fine-tune attack (locked or unlocked init)
│   ├── run.sh                       # Dispatcher: run.sh <lock|finetune> <config.yaml>
│   ├── run_finetune_v8_slurm.sh     # SLURM: v8 locked + unlocked finetune in one job
│   ├── run_pipeline.sh              # SLURM: v9 full pipeline (lock + 2× finetune)
│   ├── run_lock_slurm.sh            # SLURM: standalone lock job
│   ├── eval_pretrained.py           # Pretrained baseline evaluation
│   └── test_pipeline.sh             # Offline tests (no GPU needed)
├── src/
│   └── utils.py                     # Shared utilities, data loading, SVD monitoring
├── data/
│   ├── attack.fasta                 # Attack dataset (15 pathogen sequences)
│   ├── retain.fasta                 # Retain dataset (GTDB + IMG/VR, gitignored)
│   └── download_scripts/
│       ├── prepare_retain.sh        # Full download + processing pipeline
│       ├── retain.py                # Python processing (4 subcommands)
│       ├── .env                     # JGI credentials (gitignored)
│       └── .env.example             # Credential template
├── results/                         # Model checkpoints + metrics (gitignored)
│   └── lock_v8_all_linear/
│       └── model_locked.pt          # v8 lock — DONE
└── README.md
```

## Config System

All hyperparameters live in `configs/` as YAML files. **Filename stem equals `run_name`** — results land in `results/{run_name}/` automatically (no `results_dir` field needed). Omit `locked_ckpt` for an unlocked baseline run.

Config is loaded by both `lock.py` and `finetune.py` via `--config`:

```bash
python scripts/lock.py    --config configs/lock_topk5_5000steps.yaml
python scripts/finetune.py --config configs/ft_locked_topk5_5000lock_20ep.yaml
```

### Lock config fields

| Field | Example | Description |
|-------|---------|-------------|
| `run_name` | `lock_topk5_5000steps` | Also the results subdirectory name |
| `lock_steps` | `5000` | Gradient steps |
| `lock_lr` | `5e-5` | Learning rate |
| `alpha_start/end` | `0.8` → `0.3` | Linear anneal of retain weight |
| `top_k` | `5` | Number of singular values to inflate |
| `target_blocks` | `32` | Number of leading blocks to lock |
| `target_layer_patterns` | `.projections.weight`, ... | Layer name suffixes to target |

### Finetune config fields

| Field | Example | Description |
|-------|---------|-------------|
| `run_name` | `ft_locked_topk5_5000lock_20ep` | Results subdirectory |
| `train_steps` | `33600` | ~40 epochs over attack.fasta |
| `lr` | `5e-5` | AdamW learning rate |
| `locked_ckpt` | `results/lock_topk5_5000steps/model_locked.pt` | Omit for unlocked baseline |
| `target_blocks` | `32` | Blocks to fine-tune |

## Requirements

### Python environment

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/evo-design/evo.git
pip install matplotlib tqdm pyyaml
```

Python 3.10+ required. CUDA with bfloat16 support recommended (A100/H100).

### Data download (CPU-only, no pip needed)

Standard tools: `curl wget python3 awk zcat unzip`

JGI account required for IMG/VR data — register at https://contacts.jgi.doe.gov/registration/new

## Data Setup

### Build retain.fasta

The retain set is built from GTDB r220 (bacteria/archaea) + IMG/VR v4 (prokaryotic phages) — ~6,600 sequences, ~2 GB.

```bash
cp data/download_scripts/.env.example data/download_scripts/.env
# Edit .env: fill in JGI_USER and JGI_PASS

bash data/download_scripts/prepare_retain.sh   # 1–2 h, use screen/tmux
```

All steps are idempotent — re-running skips completed steps. Final `data/retain.fasta` is a symlink to `$SCRATCH_DIR/retain.fasta`.

## Running Experiments

### Dispatcher: run.sh

All jobs route through `scripts/run.sh`, which auto-detects GPU count and uses `torchrun` for multi-GPU:

```bash
bash scripts/run.sh lock     configs/lock_topk5_5000steps.yaml
bash scripts/run.sh finetune configs/ft_locked_topk5_5000lock_20ep.yaml
```

### V8 finetune comparison (checkpoint already exists)

Runs both locked and unlocked finetune sequentially in one SLURM job:

```bash
sbatch scripts/run_finetune_v8_slurm.sh
```

- **Stage 1**: `ft_locked_v8_1000lock_topk5_20ep.yaml` (loads `results/lock_v8_all_linear/model_locked.pt`)
- **Stage 2**: `ft_unlocked_v8_1000lock_topk5_20ep.yaml` (pretrained baseline)
- Prints val_loss gap summary at end

### V9 full pipeline (lock + 2× finetune)

```bash
sbatch scripts/run_pipeline.sh            # run all three stages
sbatch scripts/run_pipeline.sh --skip-lock  # skip Stage 1 if checkpoint exists
```

- **Stage 1**: lock → `configs/lock_topk5_5000steps.yaml`
- **Stage 2**: ft locked → `configs/ft_locked_topk5_5000lock_20ep.yaml`
- **Stage 3**: ft unlocked → `configs/ft_unlocked_topk5_5000lock_20ep.yaml`

### Standalone lock

```bash
sbatch scripts/run_lock_slurm.sh          # single or multi-GPU via SLURM
bash  scripts/run.sh lock configs/lock_topk5_5000steps.yaml   # interactive
```

### Tests (no GPU)

```bash
bash scripts/test_pipeline.sh
```

Covers: YAML parse, config values, checkpoint paths, epoch math, `_load_config` wiring, `--skip-lock` guard, required files, attack.fasta content.

## Experiment Configs at a Glance

| Config | Type | Steps | top_k | locked_ckpt | Status |
|--------|------|-------|-------|-------------|--------|
| `lock_v8_all_linear` | lock | 1000 | 5 | — | **DONE** |
| `lock_topk5_5000steps` | lock | 5000 | 5 | — | pending |
| `ft_locked_v8_1000lock_topk5_20ep` | finetune | 33 600 | — | lock_v8_all_linear | pending |
| `ft_unlocked_v8_1000lock_topk5_20ep` | finetune | 33 600 | — | none | pending |
| `ft_locked_topk5_5000lock_20ep` | finetune | 33 600 | — | lock_topk5_5000steps | pending |
| `ft_unlocked_topk5_5000lock_20ep` | finetune | 33 600 | — | none | pending |

## Evo Architecture Notes

Evo-1-8k-base is a StripedHyena model:
- 32 blocks total: 29 Hyena (SSM) + 3 Attention (at layers 8, 16, 24)
- `hidden_size=4096`, `vocab_size=512`, `max_seq_len=8192`
- Each Hyena block: projections → SSM filter → out_filter_dense → MLP (SwiGLU)
- SSM filter parameters (poles, residues, short_filter) are **not** locked — they are ~0.03% of block params
- Locked layers per block: `projections.weight`, `out_filter_dense.weight`, `mlp.l1/l2/l3.weight` (5 matrices × 32 blocks = 160 matrices total)

## Locking Procedure

- Loads `data/retain.fasta` (bacteria/archaea/phage, not task-specific)
- Loss: $\mathcal{L}_\text{lock} = \alpha \cdot \mathcal{L}_\text{retain} - (1-\alpha) \cdot \overline{\sigma_1}$
- Alpha anneals linearly 0.8 → 0.3 over `lock_steps` (preserve utility early, maximize inflation late)
- Spectral term uses **randomized SVD** (`torch.svd_lowrank`) — on-GPU, differentiable, ~50× faster than full SVD
- DDP: spectral backward runs inside `no_sync()` (weights identical across ranks); retain backward syncs normally
- Saves to `results/{run_name}/model_locked.pt`

## Key Changes from v7

| Aspect | v7 | Current |
|--------|----|---------|
| Config format | Hardcoded `CONFIG` block in Python | YAML files in `configs/`, `--config` CLI arg |
| Lock targets | `projections.weight` only (8 matrices) | All `nn.Linear` in all 32 blocks (160 matrices) |
| Spectral aggregation | `.sum()` | `.mean()` (scale-stable) |
| Lock steps | 50 | 1000–5000 |
| top_k | 1 | 5 |
| Seq len (lock) | 64 | 1024 |
| Seq len (fine-tune) | 128 | 1024 |
| SVD method | Full `svdvals` on GPU | Randomized `svd_lowrank` (no OOM, 50× faster) |
| Alpha schedule | Fixed 0.5 | Linear 0.8 → 0.3 |
| Multi-GPU | None | DDP via torchrun, auto-detected in `run.sh` |
| Retain dataset | None | GTDB r220 + IMG/VR v4 (6,600 seqs, ~2 GB) |
| Data split | Same data for lock + attack | Separate retain/attack datasets |

