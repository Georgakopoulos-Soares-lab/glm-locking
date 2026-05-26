# glm-locking

Reproducible pipeline for:

> **Spectral locking as a defence for open-weight genomic foundation models**  
> Aris Karatzikos · Aggeliki Vasilopoulou · Candace SY Chan · Ioannis Mouratidis · Ilias Georgakopoulos-Soares  
> *Bioinformatics* (2026)

We apply Spectral Deformation (SpecDef) locking ([Rosati et al. 2025/2026](https://arxiv.org/abs/2406.00954))
to [Evo-1-8k-base](https://huggingface.co/togethercomputer/evo-1-8k-base) (7 B parameters, StripedHyena)
and stress-test it against fifteen attack configurations spanning five attack classes. Every practical
attack is deterred; only the SVD-chain factorisation (Theorem 8) achieves partial recovery, constrained
to a k-dependent PPL–AUROC trade-off surface from which the unlocked operating point is unreachable.
Raising α ten-fold consistently shifts the naive-FT attacker toward the defender; at α=3×10⁵ the
standard-rate attack no longer exceeds the unlocked AUROC ceiling.

---

## Findings at a glance

Results from Table 1 of the paper. Four lock strengths (α ∈ {10⁴, 3×10⁴, 10⁵, 3×10⁵}) are
evaluated; attacks at α = 3×10⁴ are labelled the "primary lock" and are the primary evaluation.
Virological capability is measured by mean AUROC over three HVUE tasks (Host Tropism,
Pathogenicity, Transmissibility) using ℓ₂-regularised linear SVM probes on mean-pooled
final-layer activations.

| Condition | α | PPL ↓ | Tropism | Pathog. | Trans. | Mean AUROC ↑ |
|---|---|---|---|---|---|---|
| Pretrained (no FT) | — | 3.729 | 0.860 | 0.815 | 0.852 | 0.842 |
| Locked, no FT | 3×10⁴ | 3.729 | ≡ pretrained | | | |
| Unlocked FT (25k steps) | — | 3.487 | 0.876 | 0.842 | 0.871 | **0.867** |
| **A.** Naive FT (η=10⁻⁶, prescribed) | 10⁴ | 3.753 | 0.864 | 0.818 | 0.862 | 0.848 |
| **B.** Naive FT (η=10⁻⁶, prescribed) | 3×10⁴ | 3.816*** | 0.797 | 0.788 | 0.785 | 0.790*** |
| **C′.** Naive FT (η=10⁻⁵, standard) | 3×10⁴ | 3.776** | **0.905** | **0.847** | **0.894** | **0.882***\* |
| **C.** Naive FT (η=10⁻⁴, aggressive) | 3×10⁴ | *diverged — grad norm = ∞ from step 100* | | | | |
| **D.** LoRA r=16 | 10⁴ | 3.729 | 0.839 | 0.766 | 0.779 | 0.795** |
| **E.** Inserted-layer bypass | 10⁴ | 4.091*** | 0.688 | 0.735 | 0.845 | 0.756*** |
| **F.** Inserted-layer bypass | 3×10⁴ | 4.123*** | 0.707 | 0.735 | 0.848 | 0.763*** |
| **G.** SVD-chain k=3 | 10⁴ | 3.667* | 0.817 | 0.782 | 0.819 | 0.806** |
| **H.** SVD-chain k=2 | 3×10⁴ | 3.745* | 0.845 | 0.786 | 0.840 | 0.824** |
| **I.** SVD-chain k=3 | 3×10⁴ | 3.705** | 0.858 | 0.803 | 0.845 | 0.836** |
| **J.** SVD-chain k=5 | 3×10⁴ | 3.760* | 0.852 | 0.808 | 0.845 | 0.835** |
| **K.** Naive FT (η=10⁻⁵, standard) | 10⁵ | 3.798 | 0.879 | 0.839 | 0.871 | 0.863* |
| **L.** Naive FT (η=10⁻⁶, prescribed) | 10⁵ | 3.876 | 0.817 | 0.784 | 0.823 | 0.808** |
| **M.** Naive FT (η=10⁻⁵, standard) | 3×10⁵ | 3.810 | 0.872 | 0.831 | 0.871 | 0.858 |
| **N.** Naive FT (η=10⁻⁶, prescribed) | 3×10⁵ | 5.861*** | 0.834 | 0.806 | 0.830 | 0.823 |

p-values by paired bootstrap (n=2000) vs. pretrained: \*p<0.05; \*\*p<0.01; \*\*\*p<10⁻³.  
PPL p-values for K–M not computable from retained logs (per-batch losses not retained);  
raw values are unambiguously higher than pretrained.

Key findings:
- **Naive FT at the standard rate (C′)** reaches the highest AUROC of any condition (0.882, above the unlocked ceiling) but PPL remains above pretrained (3.776 vs 3.487) — a joint-metric failure.
- Under the **primary lock** (α=3×10⁴), the prescribed-rate naive attacker (B) has AUROC **actively suppressed** below pretrained (0.790 vs 0.842).
- **Aggressive fine-tuning** (C) hard-diverges — the curvature barrier is a hard arithmetic constraint, not a soft penalty.
- The **inserted-layer bypass** (E, F) catastrophically overfits, producing models worse than pretrained on both axes.
- Only the **SVD-chain factorisation** (H, I, J) breaks the curvature barrier without catastrophic overfitting, but traces a k-dependent PPL–AUROC trade-off; k=3 is the best at PPL 3.705, AUROC 0.836 — no configuration enters the target zone (PPL ≤ 3.487 AND AUROC ≥ 0.867).
- **α-scaling (K–N)**: raising α ten-fold reduces the standard-rate AUROC by ~0.024 per step; at α=3×10⁵ the standard-rate attacker (M) no longer exceeds the unlocked ceiling (0.858 < 0.867).

---

## Repo layout

```
configs/
  lock/
    alpha10k.yaml              # lock α=10⁴  — produces results/lock_alpha10k/model_specdef.pt
    alpha30k.yaml              # lock α=3×10⁴ — produces results/lock_alpha30k/model_specdef.pt
    alpha100k.yaml             # lock α=10⁵  — produces results/lock_alpha100k/model_specdef.pt
    alpha300k.yaml             # lock α=3×10⁵ — produces results/lock_alpha300k/model_specdef.pt
  finetune/
    unlocked_25k_v2.yaml          # unlocked baseline  (η=1e-5, 25k steps)
    locked_a10k_lr1e6_25k.yaml    # Attack A:  naive FT, α=10⁴,   η=1e-6
    locked_a30k_lr1e6_25k.yaml    # Attack B:  naive FT, α=3×10⁴, η=1e-6 (prescribed)
    locked_a30k_lr1e5_25k.yaml    # Attack C′: naive FT, α=3×10⁴, η=1e-5 (standard rate)
    locked_a10k_lr1e4.yaml        # Attack C:  aggressive FT, α=3×10⁴, η=1e-4 (diverges)
    lora_a10k_25k.yaml            # Attack D:  LoRA r=16, α=10⁴
    bypass_a10k_25k_v2.yaml       # Attack E:  inserted-layer bypass, α=10⁴
    bypass_a30k_25k.yaml          # Attack F:  inserted-layer bypass, α=3×10⁴
    theorem8_a10k_k3_25k.yaml     # Attack G:  SVD-chain k=3, α=10⁴
    theorem8_a30k_k2_25k.yaml     # Attack H:  SVD-chain k=2, α=3×10⁴
    theorem8_a30k_k3_25k.yaml     # Attack I:  SVD-chain k=3, α=3×10⁴
    theorem8_a30k_k5_25k.yaml     # Attack J:  SVD-chain k=5, α=3×10⁴
    locked_a100k_lr1e5_25k.yaml   # Attack K:  naive FT, α=10⁵,   η=1e-5 (standard)
    locked_a100k_lr1e6_25k.yaml   # Attack L:  naive FT, α=10⁵,   η=1e-6 (prescribed)
    locked_a300k_lr1e5_25k.yaml   # Attack M:  naive FT, α=3×10⁵, η=1e-5 (standard)
    locked_a300k_lr1e6_25k.yaml   # Attack N:  naive FT, α=3×10⁵, η=1e-6 (prescribed)
data/
  download_scripts/            # data download and preparation scripts (see §2)
src/
  utils.py                     # shared data-loading, model-loading, SpecDef inject helpers
  lora.py                      # LoRA injection (skips SpecDefLinear layers)
scripts/
  lock_specdef.py              # Step 1  — algebraic SVD-based SpecDef locking
  finetune.py                  # Step 2  — fine-tune locked or unlocked checkpoint
  split_attack_fasta.py        # build accession-disjoint genus-stratified train/heldout split
  attack_ppl.py                # Step 3a — compute held-out perplexity across all checkpoints
  hvue_extract_one_ckpt.py     # Step 3b — extract mean-pooled final-layer embeddings
  hvue_probe.py                # Step 4  — ℓ₂-SVM linear probe → AUROC / accuracy / F1 / MCC
  hvue_significance.py         # Step 5  — paired bootstrap significance vs pretrained baseline
  make_paper_figures.py        # regenerate all paper figures from results/ CSVs
  precision_diag.py            # fp32/bf16 forward-pass preservation diagnostic
  profile_attack_overhead.py   # measure per-step time and parameter counts per config
  run_pipeline.sh              # end-to-end convenience wrapper (Steps 2–5)
```

---

## 1. Environment setup

```bash
bash setup_evo_env.sh
# Creates conda env 'evo' with:
#   Python 3.10, PyTorch 2.7.0 (CUDA 12.8), FlashAttention 2.7.4,
#   evo-model, biopython, pandas, scipy, matplotlib, seaborn

conda activate evo
```

`setup_evo_env.sh` installs the `evo-model` package which pulls Evo-1-8k-base weights
from HuggingFace on first use. No manual weight download is required. Set
`HF_HOME` to a path with sufficient disk space (~30 GB) before running if the default
HuggingFace cache directory is on a small filesystem.

All scripts are designed to be run from the repo root directory.

---

## 2. Data preparation

### 2a. Attack corpus

The attack corpus consists of 910 human-pathogenic virus assemblies sourced from NCBI Virus,
covering Influenza A, SARS-CoV-2, Ebola, HIV, Dengue, West Nile, and related pathogens.

```bash
# Download all 910 assemblies from NCBI Virus  (~3.4 GB, requires internet)
bash data/download_scripts/download_attack.sh
# Output: data/attack.fasta

# Genus-stratified accession-disjoint train / held-out split
python scripts/split_attack_fasta.py
# Output: data/attack_train.fasta   (544 assemblies, ~25 Mnt)
#         data/attack_heldout.fasta (366 assemblies, ~16.8 Mnt)
```

The split is genus-stratified so that the held-out set tests generalisation across genera,
not just within-genus interpolation. Split reproducibility is fixed by `--seed 42` (default).

### 2b. HVUE downstream benchmark

The HVUE benchmark ([Dutta et al. 2026](https://huggingface.co/datasets/duttaprat/HVUE))
provides three binary classification tasks: Host Tropism, Pathogenicity, Transmissibility.
Each task has a training (n=3000) and validation (n=2000) split.

```bash
python -c "
from datasets import load_dataset
for task in ['Host_Tropism', 'Pathogenicity', 'Transmissibility']:
    load_dataset('duttaprat/HVUE', task).save_to_disk(f'data/hvue/{task}')
"
# Output: data/hvue/{Host_Tropism,Pathogenicity,Transmissibility}/
```

---

## 3. Locking

Apply SpecDef to Evo-1-8k-base, producing four locked checkpoints at different lock strengths.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/lock_specdef.py configs/lock/alpha10k.yaml
# Output: results/lock_alpha10k/model_specdef.pt   (weaker lock, used for Attacks A, D, E, G)

CUDA_VISIBLE_DEVICES=0 python scripts/lock_specdef.py configs/lock/alpha30k.yaml
# Output: results/lock_alpha30k/model_specdef.pt   (primary lock, used for Attacks B, C, C′, F, H, I, J)

CUDA_VISIBLE_DEVICES=0 python scripts/lock_specdef.py configs/lock/alpha100k.yaml
# Output: results/lock_alpha100k/model_specdef.pt  (used for Attacks K, L)

CUDA_VISIBLE_DEVICES=0 python scripts/lock_specdef.py configs/lock/alpha300k.yaml
# Output: results/lock_alpha300k/model_specdef.pt  (used for Attacks M, N)
```

**What locking does:** For each of the 32 write-side output projections (4096×4096), SpecDef
computes the thin SVD W = UΣVᵀ, inflates the top 25 singular values by α (W̃ = U·diag(α·σ₁,...,
σ₂₆,...·σᵣ)·Vᵀ), and constructs a compensation matrix C = U·(Σ·Σ̃⁻¹)·Uᵀ that exactly restores
the forward pass: C·W̃·x = W·x. The inflated singular values raise the Hessian curvature along
the dominant weight directions by α², forcing stable fine-tuning to use η ≤ 1/(α·σ₁)². For
α=3×10⁴, this reduces the unlocked stable learning rate (10⁻⁵) to ≤10⁻⁶. The forward-pass
drift after locking is verified to be <3×10⁻⁴ on the held-out corpus.

---

## 4. Fine-tuning (attack runs)

All attack runs use the same shared training protocol: 8-bit AdamW (weight decay 0),
gradient clipping to norm 1.0, fp32 precision, 1024-nt windows with 512-nt stride, batch
size 1, gradient-accumulation steps 4, 25 000 steps, on a single NVIDIA A100 80 GB.
This amounts to 102.4 M tokens per run (25 000 × 4 × 1024).

The convenience wrapper `run_pipeline.sh` executes fine-tuning (Step 2) followed by
PPL evaluation (Step 3a), HVUE embedding extraction (Step 3b), linear probe (Step 4),
and bootstrap significance (Step 5) in one command:

```bash
# Usage: bash scripts/run_pipeline.sh <GPU_ID> <CONFIG_YAML>

# Reference: unlocked fine-tuning ceiling
bash scripts/run_pipeline.sh 0 configs/finetune/unlocked_25k_v2.yaml

# Attack A — Naive FT, α=10⁴, η=1e-6
bash scripts/run_pipeline.sh 0 configs/finetune/locked_a10k_lr1e6_25k.yaml

# Attack B — Naive FT, α=3×10⁴, η=1e-6
bash scripts/run_pipeline.sh 0 configs/finetune/locked_a30k_lr1e6_25k.yaml

# Attack C′ — Naive FT, α=3×10⁴, η=1e-5 (standard rate — highest AUROC of any condition)
bash scripts/run_pipeline.sh 0 configs/finetune/locked_a30k_lr1e5_25k.yaml

# Attack C — Aggressive FT, α=3×10⁴, η=1e-4  (diverges at step ~100, no checkpoint produced)
bash scripts/run_pipeline.sh 0 configs/finetune/locked_a10k_lr1e4.yaml

# Attack D — LoRA r=16, α=10⁴
bash scripts/run_pipeline.sh 0 configs/finetune/lora_a10k_25k.yaml

# Attack E — Inserted-layer bypass, α=10⁴
bash scripts/run_pipeline.sh 0 configs/finetune/bypass_a10k_25k_v2.yaml

# Attack F — Inserted-layer bypass, α=3×10⁴
bash scripts/run_pipeline.sh 0 configs/finetune/bypass_a30k_25k.yaml

# Attack G — SVD-chain k=3, α=10⁴
bash scripts/run_pipeline.sh 0 configs/finetune/theorem8_a10k_k3_25k.yaml

# Attack H — SVD-chain k=2, α=3×10⁴
bash scripts/run_pipeline.sh 0 configs/finetune/theorem8_a30k_k2_25k.yaml

# Attack I — SVD-chain k=3, α=3×10⁴
bash scripts/run_pipeline.sh 0 configs/finetune/theorem8_a30k_k3_25k.yaml

# Attack J — SVD-chain k=5, α=3×10⁴
bash scripts/run_pipeline.sh 0 configs/finetune/theorem8_a30k_k5_25k.yaml

# Lock-strength robustness (α-scaling, naive FT only)
# Attack K — Naive FT, α=10⁵, η=1e-5 (standard rate)
bash scripts/run_pipeline.sh 0 configs/finetune/locked_a100k_lr1e5_25k.yaml

# Attack L — Naive FT, α=10⁵, η=1e-6 (prescribed)
bash scripts/run_pipeline.sh 0 configs/finetune/locked_a100k_lr1e6_25k.yaml

# Attack M — Naive FT, α=3×10⁵, η=1e-5 (standard rate — first below unlocked AUROC ceiling)
bash scripts/run_pipeline.sh 0 configs/finetune/locked_a300k_lr1e5_25k.yaml

# Attack N — Naive FT, α=3×10⁵, η=1e-6 (prescribed — simultaneous PPL+AUROC collapse)
bash scripts/run_pipeline.sh 0 configs/finetune/locked_a300k_lr1e6_25k.yaml
```

Runs can be parallelised across GPUs (one config per device). Logs are written to
`logs/runs/<config_name>.log`. The best checkpoint (by validation loss) is saved to
`results/<run_name>/model_best.pt`.

**To run fine-tuning only** (skip PPL/HVUE evaluation):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --config configs/finetune/locked_a30k_lr1e6_25k.yaml
```

**Multi-GPU fine-tuning** (e.g. 4 GPUs, data-parallel):

```bash
bash scripts/run_pipeline.sh 0 configs/finetune/theorem8_a30k_k5_25k.yaml 4
```

---

## 5. Evaluation

### 5.1 Held-out perplexity

Computes held-out viral PPL for all checkpoints found in `results/`. Results are
appended to a single registry CSV so the command is safe to re-run incrementally.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/attack_ppl.py --n_batches 64
# Output: results/attack_heldout_ppl.csv  (columns: name, ckpt, val_loss, val_ppl)
```

To evaluate a specific new checkpoint without touching the full registry:

```bash
TMP=$(mktemp --suffix=.csv)
echo "name,ckpt" > "$TMP"
echo "my_run,results/my_run/model_best.pt" >> "$TMP"
CUDA_VISIBLE_DEVICES=0 python scripts/attack_ppl.py \
    --n_batches 64 --ckpts_csv "$TMP" --out results/attack_heldout_ppl.csv
```

### 5.2 HVUE embedding extraction

Extracts 4096-dim mean-pooled final-layer activations for every checkpoint × task × split
combination. (This step is the most time-consuming; ~10–15 min per checkpoint on an A100.)

```bash
# Extract embeddings for all checkpoints listed in the paper (adjust CKPT_NAME as needed)
for CKPT_NAME in pretrained \
    ft_unlocked_25k_v2_unlocked \
    ft_locked_a10k_lr1e6_25k_locked \
    ft_locked_a30k_lr1e6_25k_locked \
    ft_locked_a30k_lr1e5_25k_locked \
    ft_lora_a10k_25k_locked \
    ft_bypass_a10k_25k_v2_locked \
    ft_bypass_a30k_25k_locked \
    ft_theorem8_a10k_k3_25k_locked \
    ft_theorem8_a30k_k2_25k_locked \
    ft_theorem8_a30k_k3_25k_locked \
    ft_theorem8_a30k_k5_25k_locked \
    ft_locked_a100k_lr1e5_25k_locked \
    ft_locked_a100k_lr1e6_25k_locked \
    ft_locked_a300k_lr1e5_25k_locked \
    ft_locked_a300k_lr1e6_25k_locked; do
  CKPT_PATH="results/${CKPT_NAME}/model_best.pt"
  [[ "$CKPT_NAME" == "pretrained" ]] && CKPT_PATH="pretrained"
  CUDA_VISIBLE_DEVICES=0 python scripts/hvue_extract_one_ckpt.py \
      --ckpt_name "$CKPT_NAME" \
      --ckpt_path "$CKPT_PATH"
done
# Output: results/hvue_embeddings/<ckpt_name>_<task>_{train,validation}.npz
```

### 5.3 HVUE linear probe

Trains ℓ₂-regularised linear SVM probes (C swept over {0.01, 0.1, 1, 10} by 5-fold CV)
for each (checkpoint, task) pair and writes AUROC, accuracy, F1, and MCC.

```bash
python scripts/hvue_probe.py \
    --emb_dir results/hvue_embeddings \
    --out results/hvue_probe.csv
# Output: results/hvue_probe.csv  (columns: ckpt, task, auroc, acc, f1, mcc, ...)
```

### 5.4 Statistical significance

Paired bootstrap significance test (n=2000 resamples) comparing each checkpoint against
the pretrained baseline on each HVUE task.

```bash
python scripts/hvue_significance.py \
    --emb_dir results/hvue_embeddings \
    --n_boot 2000
# Output: printed table of p-values per (ckpt, task)
```

### 5.5 Figures

Regenerates all paper and supplement figures from the results CSVs. Outputs are written
to `paper/` as both PDF and PNG.

```bash
python scripts/make_paper_figures.py
# Outputs (in paper/):
#   fig1_scatter.{pdf,png}          — PPL vs AUROC operating-point scatter (Fig 1)
#   fig2_main_bars.{pdf,png}        — representative-conditions bar chart (Fig 2)
#   fig3_kablation.{pdf,png}        — SVD-chain k-ablation (Fig 3)
#   fig_s1_training_curves.{pdf,png} — training dynamics (Supp. Fig. S1)
```

### 5.6 Forward-pass preservation diagnostic

Verifies that the locked checkpoint is function-equivalent to pretrained by measuring
perplexity drift of the fused C·W̃ product vs. the original W.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/precision_diag.py \
    --ckpt results/lock_alpha30k/model_specdef.pt \
    --sigmas 1e-7 1e-6 1e-5 1e-4 1e-3
```

### 5.7 Computational overhead profiling

Measures per-step wall-clock time and trainable parameter counts for each configuration.
These numbers appear in Supplementary Table S1 of the paper.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_attack_overhead.py
```

---

## Reproducibility notes

| Parameter | Value |
|---|---|
| Optimizer | 8-bit AdamW (bitsandbytes) |
| Weight decay | 0 |
| Gradient clip | norm 1.0 |
| Precision | fp32 (SVD-chain factor matmuls require fp32: σ_max(W̃) ≈ α·σ_max(W) ≈ 10⁵ exceeds bf16 range of 65 504) |
| Sequence length | 1024 nt, stride 512 nt |
| Batch size | 1 sequence, grad accum 4 (effective batch = 4096 tokens) |
| Training steps | 25 000 |
| Tokens per run | 25 000 × 4 × 1024 = 102.4 M |
| LR — unlocked baseline | 1×10⁻⁵ |
| LR — naive FT (prescribed), LoRA, bypass, SVD-chain | 1×10⁻⁶ (attacks A, B, D, E, F, G, H, I, J) |
| LR — naive FT (standard rate) | 1×10⁻⁵ (attacks C′, K, M) |
| LR — naive FT (prescribed) at higher α | 1×10⁻⁶ (attacks L, N) |
| LR — aggressive FT (Attack C) | 1×10⁻⁴ (diverges at step ~100) |
| HVUE probe | n_train=3000, n_val=2000, C ∈ {0.01, 0.1, 1, 10}, seed 42 |
| Bootstrap | n=2000, paired on same validation examples |
| Hardware | 1× NVIDIA A100 80 GB |

**SpecDef locking targets:** All 32 write-side output projections of Evo-1-8k-base — the 29
`out_filter_dense` layers in Hyena blocks and the 3 `inner_mha_cls.out_proj` layers in the
multi-head attention blocks. Top-25 singular values inflated in each.

**SVD-chain initialisation:** Each factor matrix Lᵢ is initialised from the k-th approximate
matrix root of W̃, obtained by computing the full SVD of W̃ and taking Σ^(1/k). This ensures
L_k·...·L_1 ≈ W̃ at step 0, so perplexity at the start of SVD-chain training matches the
locked (not pretrained) baseline.

---

## Citation

```bibtex
@article{glm-locking-2026,
  title   = {Weight locking deters capability-recovery attacks on
             open-weight genomic foundation models},
  author  = {Karatzikos, Aris and Vasilopoulou, Aggeliki and Chan, Candace SY and
             Mouratidis, Ioannis and Georgakopoulos-Soares, Ilias},
  journal = {Bioinformatics},
  year    = {2026},
  url     = {https://github.com/Georgakopoulos-Soares-lab/glm-locking}
}
```
