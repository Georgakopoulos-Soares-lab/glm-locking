# Archive — Gradient-Based Locking (v7–v11)

This directory contains the **original gradient-based locking** implementation and associated configs from early iterations (v7 through v11). These methods were superseded by exact algebraic SpecDef locking (`scripts/lock_specdef.py` in the main repo).

## What's here

### Scripts

| Script | Description |
|---|---|
| `lock.py` | Gradient-based SpecDef locking — trains a spectral penalty via SGD on retain data. Slow (5k–10k steps, requires GPU hours). |
| `lock_direct_scale.py` | Naive baseline: directly scales weight matrices by a constant. No compensation. |
| `eval_pretrained.py` | Standalone evaluation of the pretrained model on attack data. |
| `bio_eval.py` | Early bio evaluation script (superseded by `eval_bio_compare.py`). |
| `run_pipeline*.sh` | End-to-end pipeline scripts for SLURM (lock → finetune locked → finetune unlocked). |
| `run_*_slurm.sh` | SLURM job submission wrappers. |

### Configs

| Pattern | Description |
|---|---|
| `lock_v10*.yaml`, `lock_v11*.yaml` | Gradient-based lock configs with varying hyperparameters (alpha schedules, top_k, etc.) |
| `lock_v8_all_linear.yaml` | Lock all linear layers (v8 iteration) |
| `lock_topk5_5000steps.yaml` | Gradient-based lock with top_k=5, 5000 training steps |
| `ft_attack_v10*.yaml`, `ft_attack_v11*.yaml` | Attack configs paired with gradient-based locks |
| `ft_locked_*.yaml`, `ft_unlocked_*.yaml` | v7/v8 attack configs |
| `lock_ablation_retain_only.yaml` | Ablation: lock with retain loss only (no spectral penalty) |
| `lock_direct_scale_x100.yaml` | Direct weight scaling baseline (100×) |

## Why these were abandoned

The gradient-based approach had several problems:

1. **Slow** — Required 5k–10k GPU steps to converge the spectral penalty, consuming significant compute.
2. **Approximate** — The resulting lock was never exact; reconstruction error `||C·W̃ - W||` was nonzero and varied across layers.
3. **Unstable** — The training-based alpha schedule (`alpha_start` → `alpha_end`) was sensitive to hyperparameters and often produced inconsistent results across runs.

The algebraic SpecDef approach (`lock_specdef.py`) replaced all of this: it computes the exact SVD decomposition in seconds with zero reconstruction error (`ε ≈ 1e-10` in f64), requires no GPU training, and is fully deterministic.

## How to use (not recommended)

```bash
# These scripts expect the old lock.py interface
CUDA_VISIBLE_DEVICES=0 conda run -n evo --no-capture-output \
    python -u scripts/lock.py --config archive/configs/lock_v10_full.yaml
```

Note: `lock.py` has been moved to `archive/scripts/`, so the import path would need adjustment.
