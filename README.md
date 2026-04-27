# evo-locking

Reproducible pipeline for **"Locking Evo: spectral deformation deters naive fine-tuning but is trivially bypassed"**.

We apply SpecDef capability-locking ([Rosati et al. 2025/2026](https://arxiv.org/abs/2406.00954))
to Evo-1-8k-base and test it against (i) naive full-parameter fine-tuning and
(ii) the Theorem-8 black-box layer-injection bypass.
Key findings: the lock collapses HVUE virological AUROC by 0.18–0.32 while preserving
next-token perplexity; the bypass restores the unlocked-FT operating point
using one identity-initialised matrix per locked projection, constructable in seconds on CPU.

## Findings at a glance

| Checkpoint | PPL (↓) | Mean AUROC (↑) | Steps | s/step | Peak VRAM |
|---|---|---|---|---|---|
| Pretrained | 3.729 | 0.842 | — | — | — |
| Unlocked FT | 3.480 | 0.848 | 25 000 | 3.03 | ~46 GB |
| Locked FT α=10⁴ | 3.996 | 0.667 | 1 500 | 5.17 | ~46 GB |
| Locked FT α=10⁶ | 4.009 | 0.520 | 1 500 | 5.17 | ~46 GB |
| Bypass B-only α=10⁴ | ~3.48 | ~0.84 | 25 000 | 2.51 | ~18 GB |
| **Bypass full α=10⁴** | ~3.48 | ~0.84 | 25 000 | ~2.8 | ~47 GB |

Locked FT is ~70% slower per step than unlocked (backprop through the ill-conditioned
compensation matrix C). The bypass routes around C entirely; B-only trains only the 32
bypass matrices (537 M params, 7.1% of Evo); the full bypass trains all parameters
except the frozen W̃ and C, matching the unlocked parameter count.

## Repo layout

```
configs/
  lock/                        # one YAML per α: alpha10k, alpha30k, alpha100k, alpha1M
  finetune/
    unlocked_25k.yaml          # baseline: regular FT, 25 000 steps
    locked_a{10k,30k,100k}_lr1e6_25k.yaml   # 25 000-step locked sweep (stable LR)
    locked_a1M_lr1e7_25k.yaml               # α=10⁶, lr=1e-7
    locked_a10k_lr{1e4,3e5,3e6}.yaml        # LR-instability evidence
    locked_a*_s{123,456}.yaml               # seed-pair variance runs
    bypass_a10k_25k.yaml                    # Theorem-8 bypass, B-matrix only
    bypass_a10k_full_25k.yaml               # Theorem-8 bypass, full parameter budget
    lora_attack_a100k_5k.yaml               # LoRA-only attack (MLP/QKV only)
data/
  download_scripts/            # scripts to fetch retain.fasta + attack.fasta
  attack_train.fasta           # gitignored — build with split_attack_fasta.py
  attack_heldout.fasta         # gitignored
  hvue/                        # gitignored — parquet shards from duttaprat/HVUE
src/
  utils.py                     # shared helpers
  lora.py                      # LoRA injection (skips SpecDefLinear layers)
scripts/
  lock_specdef.py              # Step 1  — SVD-based SpecDef locking
  finetune.py                  # Step 2  — fine-tune locked or unlocked checkpoint
  split_attack_fasta.py        # build accession-disjoint train/heldout split
  attack_ppl.py                # Step 3a — held-out perplexity across all checkpoints
  hvue_extract_one_ckpt.py     # Step 3b — extract mean-pooled residual-stream embeddings
  hvue_probe.py                # Step 4  — ℓ₂-LogReg linear probe → AUROC table
  hvue_significance.py         # Step 5  — paired bootstrap vs pretrained baseline
  make_paper_figures.py        # regenerate all paper/supplement figures from results/
  precision_diag.py            # sharp-minimum perturbation diagnostic
  watch_and_launch_bypass.sh   # auto-launch full bypass run + VRAM/timing report
  run_pipeline.sh              # end-to-end convenience wrapper (Steps 1–5)
results/                       # gitignored — produced by the pipeline
logs/                          # gitignored — per-run training logs
paper/                         # gitignored — manuscript draft (LaTeX)
figures/                       # gitignored — generated PDF figures
archive/                       # gitignored — legacy exploratory scripts
```

## 1. Setup

```bash
bash setup_evo_env.sh      # creates conda env `evo` with all Evo/PyTorch/bitsandbytes deps
conda activate evo
```

## 2. Data

```bash
# Attack corpus (910 human-pathogenic virus assemblies, NCBI Virus)
bash data/download_scripts/download_attack.sh      # → data/attack.fasta
python scripts/split_attack_fasta.py               # → data/attack_train.fasta (544)
                                                   #   data/attack_heldout.fasta (366)
# Split is by NCBI accession: zero sequences shared between files.

# HVUE downstream benchmark (HuggingFace: duttaprat/HVUE)
python -c "
from datasets import load_dataset
for task in ['Host_Tropism','Pathogenicity','Transmissibility']:
    ds = load_dataset('duttaprat/HVUE', task)
    ds.save_to_disk(f'data/hvue/{task}')
"
```

## 3. Lock

```bash
for cfg in configs/lock/alpha*.yaml; do
  CUDA_VISIBLE_DEVICES=0 python scripts/lock_specdef.py "$cfg"
done
```

Produces `results/lock_alpha{10k,30k,100k,1M}/model_specdef.pt`.
Each checkpoint is function-equivalent to pretrained (C·W̃ = W to bf16 precision)
but inflates the top-25 singular values in 32 output projections by α.

## 4. Fine-tune

```bash
# Unlocked baseline (25 000 steps, lr=1e-5)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py configs/finetune/unlocked_25k.yaml

# Stable-LR locked sweep (lr ∝ 1/α, 25 000 steps)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py configs/finetune/locked_a10k_lr1e6_25k.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/finetune.py configs/finetune/locked_a30k_lr1e6_25k.yaml
CUDA_VISIBLE_DEVICES=2 python scripts/finetune.py configs/finetune/locked_a100k_lr1e6_25k.yaml
CUDA_VISIBLE_DEVICES=3 python scripts/finetune.py configs/finetune/locked_a1M_lr1e7_25k.yaml

# Theorem-8 bypass (full parameter budget, 25 000 steps, lr=1e-5)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py configs/finetune/bypass_a10k_full_25k.yaml
```

All configs train on `data/attack_train.fasta`, validate on `data/attack_heldout.fasta`.
Outputs go to `results/<run_name>/model_best.pt`.

The watcher script auto-launches the full bypass run when a GPU frees up and collects a
timing + VRAM report after 500 steps:
```bash
bash scripts/watch_and_launch_bypass.sh --report-after=500
```

## 5. Evaluate

### 5.1 Held-out perplexity

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/attack_ppl.py --n_batches 64
# → results/attack_heldout_ppl.csv  (name, ckpt, val_loss, val_ppl)
```

### 5.2 HVUE linear probe

```bash
# Extract 4096-dim mean-pooled residual-stream embeddings for every ckpt × task × split
for ckpt in pretrained \
            results/ft_unlocked_25k/model_best.pt \
            results/ft_locked_a10k_lr1e6_25k/model_best.pt \
            results/ft_locked_a30k_lr1e6_25k/model_best.pt \
            results/ft_locked_a100k_lr1e6_25k/model_best.pt \
            results/ft_locked_a1M_lr1e7_25k/model_best.pt \
            results/ft_bypass_a10k_full_25k/model_best.pt; do
  for task in Host_Tropism Pathogenicity Transmissibility; do
    for split in train validation; do
      CUDA_VISIBLE_DEVICES=0 python scripts/hvue_extract_one_ckpt.py \
        --ckpt "$ckpt" --task "$task" --split "$split" \
        --out results/hvue_embeddings/
    done
  done
done

# ℓ₂-regularised logistic regression probe (C ∈ {0.01,0.1,1,10}, val AUROC)
python scripts/hvue_probe.py \
  --emb_dir results/hvue_embeddings \
  --out results/hvue_probe.csv

# Paired bootstrap significance (n=2000) vs pretrained baseline
python scripts/hvue_significance.py \
  --probe_csv results/hvue_probe.csv \
  --out results/hvue_significance.csv
```

### 5.3 Figures

```bash
python scripts/make_paper_figures.py
# writes: figures/fig1_main.pdf, fig2_bypass.pdf, fig_s1_training_curves.pdf
```

### 5.4 Sharp-minimum sensitivity diagnostic

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/precision_diag.py \
  --ckpt results/lock_alpha100k/model_specdef.pt \
  --sigmas 1e-7 1e-6 1e-5 1e-4 1e-3
```

## Reproducibility notes

- All FT runs: 8-bit AdamW, weight decay 0, gradient clip 1.0, bfloat16, batch 1, grad accum 4, seq len 1024, no warmup or decay, gradient checkpointing.
- Stable LR per α: 1e-6 for α∈{10⁴,3×10⁴,10⁵}; 1e-7 for α=10⁶ (follows η∝1/σ_max scaling from Rosati et al.).
- HVUE probe: n_train=3 000, n_val=2 000, fixed seed 42. Probe C selected on val AUROC.
- All bootstrap tests: n=2 000 resamples, paired on the same validation examples.
- Hardware: NVIDIA A100 80 GB SXM. Locked FT ≈ 5.17 s/step; unlocked FT ≈ 3.03 s/step; bypass full ≈ 2.8 s/step.

## Citation

```bibtex
@article{evo-locking-2026,
  title   = {Locking Evo: spectral deformation deters naive fine-tuning but is trivially bypassed},
  author  = {Anonymous Authors},
  year    = {2026},
  note    = {Preprint}
}
```
