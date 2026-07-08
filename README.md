# glm-locking

> **Safeguarding open-weight genomic foundation models through weight locking**
> Aris Karatzikos · Aggeliki Vasilopoulou · Candace SY Chan · Ioannis Mouratidis · Ilias Georgakopoulos-Soares

We apply Spectral Deformation (SpecDef) weight-locking
([Rosati et al.](https://arxiv.org/abs/2406.00954)) to
[Evo-1-8k-base](https://huggingface.co/togethercomputer/evo-1-8k-base) (7 B params)
and test whether capability-recovery attacks can fine-tune virological capability back in.
Capability is measured by **LoRA fine-tuning** on three HVUE tasks — not frozen-feature probing,
which we show is confounded by sequence composition.

**Result in one line:** the lock defends against the naive attacker (capability driven *below*
pretrained, or left inert), while the informed SVD-chain attacker recovers it only at 1.6–2.5× the
compute and up to +23.6 GB of GPU memory.

---

## Results

### Main result — downstream HVUE capability under LoRA fine-tuning

Mean AUROC / MCC across three seeds. The *k*-mer row is a 4-mer-frequency logistic-regression
baseline (train 3000 / eval 2000). Significance by paired bootstrap (*n* = 10 000) vs. pretrained
(\*p<0.05, \*\*p<0.01, \*\*\*p<10⁻³).

| Checkpoint | Host Tropism | Pathogenicity | Transmissibility |
|---|---|---|---|
| *k*-mer baseline | 0.903 / 0.662 | 0.846 / 0.549 | 0.914 / 0.733 |
| **Pretrained** | 0.933 / 0.751 | 0.957 / 0.809 | 0.932 / 0.765 |
| **Locked, no FT** | 0.938 / 0.744 | 0.954 / 0.792 | 0.943 / 0.774 |
| **Unlocked FT** *(attacker ceiling)* | 0.937 / 0.752 | 0.973 / 0.867 \*\*\* | 0.949 / 0.796 \*\* |
| Naive — Full FT, strong lock (**M**) | 0.914 / 0.698 \*\*\* | 0.938 / 0.744 \*\*\* | 0.934 / 0.777 |
| Naive — LoRA-locking | 0.935 / 0.744 | 0.952 / 0.798 | 0.930 / 0.761 |
| Informed — SVD-chain **k=2** | 0.923 / 0.708 | 0.980 / 0.882 \*\*\* | 0.948 / 0.779 |
| Informed — SVD-chain **k=3** | 0.931 / 0.746 | 0.975 / 0.863 \*\*\* | 0.949 / 0.782 |

- **Naive full FT under the strong lock (M)** is significantly *below* pretrained on Pathogenicity
  and Host Tropism — the attack becomes a capability **loss**.
- **Naive LoRA-locking** recovers nothing (sits at pretrained on every task).
- **Informed SVD-chain (k=2/k=3)** recovers Pathogenicity to the unlocked level — but at a cost (below).

### Held-out viral perplexity (main conditions)

PPL is a language-modelling diagnostic only — it does **not** track recovered capability (the
SVD-chain recovers capability while PPL stays near pretrained; LoRA-locking leaves PPL unchanged yet
recovers nothing).

| Condition | α | PPL ↓ |
|---|---|---|
| Pretrained / Locked, no FT | 3×10⁴ | 3.729 |
| Unlocked FT *(target)* | — | **3.487** |
| Naive Full FT (M) | 3×10⁵ | 3.810 |
| LoRA-locking | 10⁴ | 3.729 |
| SVD-chain k=2 / k=3 | 3×10⁴ | 3.745 / 3.705 |

The full 15-condition PPL table (A–N) and the compute/memory overhead table are in the manuscript
([paper/main.tex](paper/main.tex), Tables S1 and S2). Informed-attacker overhead vs. unlocked FT:
**k=2 1.60×** (+8.1 GB), **k=3 1.91×** (+12.9 GB), **k=5 2.51×** (+23.6 GB) per step.

---

## Pipeline

Run the steps in order — each step's output feeds the next. All commands run from the repo root.
Terminology: **primary lock** = α=3×10⁴; **strong lock** = α=3×10⁵ (produces the main naive
checkpoint M).

### 1. Environment

```bash
git clone https://github.com/Georgakopoulos-Soares-lab/glm-locking
cd glm-locking
bash setup_evo_env.sh        # conda env 'evo': py3.10, torch 2.7 (cu128), flash-attn, evo-model
conda activate evo
export HF_HOME=/path/to/big/disk/hf_cache   # Evo weights (~28 GB) auto-download here on first use
```

Verify: `python -c "from evo import Evo; print('OK')"`. Needs an A100/H100 with **80 GB**
(SVD-chain k=5 uses the full 80 GB). On quota-limited disks, symlink results to scratch:
`mkdir -p /data/$USER/results && ln -sf /data/$USER/results results` (it is gitignored).

### 2. Data — attack corpus + HVUE benchmark

```bash
# Attack corpus: 910 human-pathogen assemblies → 544 train / 366 held-out (accession-disjoint)
bash data/download_scripts/download_attack.sh      # → data/attack.fasta
python scripts/split_attack_fasta.py               # → data/attack_train.fasta, data/attack_heldout.fasta

# HVUE benchmark (three binary tasks, 3000 train / 2000 val each)
python -c "
from datasets import load_dataset
for t in ['Host_Tropism','Pathogenecity','Transmissibility']:
    load_dataset('duttaprat/HVUE', t).save_to_disk(f'data/hvue/{t}')
"
```

> The HVUE pathogenicity task key is spelled `Pathogenecity` in the dataset and code.

### 3. Lock the model

Pure SVD operation (no training, seconds per layer). Produces the four locked base checkpoints used
by the attacks:

```bash
for a in 10k 30k 100k 300k; do
  CUDA_VISIBLE_DEVICES=0 python scripts/lock_specdef.py --config configs/lock/alpha${a}.yaml
done
# → results/lock_alpha{10k,30k,100k,300k}/model_specdef.pt
```

For each of the 32 write-side output projections (4096×4096), SpecDef inflates the top-25 singular
values by α and inserts a compensation matrix `C` so the forward pass is exact (`C·W̃·x = W·x`,
PPL drift <3×10⁻⁴). The inflated spectrum forces stable fine-tuning to η ≲ 1/(α·σ₁)².

### 4. Fine-tune the attack (25 000 steps)

Each run is 25k steps of next-token prediction on the viral corpus (8-bit AdamW, grad-clip 1.0,
fp32, 1024-nt windows, batch 1 × grad-accum 4), ≈36–57 GPU-h on one A100.

```bash
# Usage: python scripts/finetune.py --config <CONFIG>   (or: run_pipeline.sh <GPU> <CONFIG> for FT + PPL)

# --- the main checkpoints ---
# Unlocked FT (attacker ceiling, η=1e-5)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py --config configs/finetune/unlocked_full910_25k.yaml

# M — naive full FT under the strong lock (α=3e5, η=1e-5: the default rate a practitioner would use)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py --config configs/finetune/locked_a300k_lr1e5_25k.yaml

# Naive LoRA-locking (α=1e4)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py --config configs/finetune/lora_a10k_25k.yaml

# Informed SVD-chain attacks (α=3e4)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py --config configs/finetune/theorem8_a30k_k2_25k.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py --config configs/finetune/theorem8_a30k_k3_25k.yaml
```

The full attack panel (A–N: naive FT across learning rates, bypass baseline, SVD k=5, and the
α-scaling robustness runs) lives in [configs/finetune/](configs/finetune/). Trained checkpoints are
placed under `checkpoints/` (see the registry below).

Held-out perplexity for any checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/attack_ppl.py --n_batches 64   # → results/attack_heldout_ppl.csv
```

### 5. LoRA fine-tune on HVUE — the capability eval

Inject rank-16 LoRA (α_LoRA=32) + a binary head, fine-tune ≤5000 steps per HVUE task, sweep
LR ∈ {1e-4, 5e-5, 1e-5}, keep best-by-val-AUROC. Run three seeds (0, 1, 42).

```bash
for s in 0 1 42; do
  CUDA_VISIBLE_DEVICES=0 python scripts/hvue_lora_finetune.py \
    --tasks Host_Tropism Pathogenecity Transmissibility \
    --ckpts pretrained locked_no_ft unlocked_ft M_a300k ft_lora_a10k svd_k2_a30k svd_k3_a30k \
    --lrs 1e-4 5e-5 1e-5 --lora_config full --seed $s \
    --out results/hvue_lora_shards/run_s${s}.csv
done
# Predictions → results/hvue_lora_preds/*.npz ;  significance (paired bootstrap n=10000):
python scripts/lora_bootstrap_analysis.py
```

`--ckpts` keys resolve through the registry below. This produces the **Main result** table above.

### 6. Figures

```bash
python scripts/build_paper_figures.py
# → paper/{fig1_lora_bars, fig2_main_bars, fig3_kablation, figS1_probing_vs_lora}.{png,pdf}
```

---

## Checkpoint registry

Checkpoints are **not distributed** — you produce them by running Steps 3–4 (locking + 25k-step
fine-tuning). Each is 13–23 GB; place them in a local `checkpoints/` directory (gitignored) under
the filenames below. `hvue_lora_finetune.py` resolves `--ckpts` keys to these paths:

| Key | Filename in `checkpoints/` | Produced by |
|---|---|---|
| `pretrained` | *(base Evo, auto-downloaded)* | — |
| `locked_no_ft` | `lock_alpha300k.pt` | Step 3 (`configs/lock/alpha300k.yaml`) |
| `unlocked_ft` | `ft_unlocked_full910_25k_unlocked.pt` | Step 4 (`unlocked_full910_25k.yaml`) |
| `M_a300k` | `ft_locked_a300k_lr1e5_25k_locked.pt` | Step 4 (`locked_a300k_lr1e5_25k.yaml`) |
| `ft_lora_a10k` | `ft_lora_a10k_25k_locked.pt` | Step 4 (`lora_a10k_25k.yaml`) |
| `svd_k2_a30k` / `svd_k3_a30k` / `svd_k5_a30k` | `ft_theorem8_a30k_k{2,3,5}_25k_locked.pt` | Step 4 (`theorem8_a30k_k{2,3,5}_25k.yaml`) |

The fine-tuning scripts write to `results/<run_name>/model_best.pt`; copy or symlink that into
`checkpoints/` under the name above (or edit the `CKPTS` map in `scripts/hvue_lora_finetune.py`).
Evo base weights come from HuggingFace — we redistribute no model weights.

---

## Reproducibility notes

| | |
|---|---|
| Locking FT | 8-bit AdamW, grad-clip 1.0, fp32, 1024-nt/stride-512, batch 1 × accum 4, 25k steps |
| LR | unlocked / standard naive = 1e-5 · prescribed / LoRA / SVD-chain = 1e-6 · aggressive = 1e-4 (diverges) |
| LoRA eval | rank 16, α_LoRA 32, ≤5000 steps, eff. batch 32, seeds 0/1/42, best LR by val AUROC |
| Significance | paired bootstrap n=10000 vs. pretrained |
| SpecDef targets | 32 write-side output projections (`out_filter_dense` ×29, `inner_mha_cls.out_proj` ×3), top k_lock=25 |
| Hardware | 1× NVIDIA A100 80 GB |

Probing scripts (`hvue_probe.py`, `hvue_extract_one_ckpt.py`) are kept **only** to reproduce
Supplementary Fig. S1 (the composition-confound figure) — do not use them for capability numbers.
Authoritative numbers live in [paper/main.tex](paper/main.tex).

---

## Citation

```bibtex
@article{glm-locking-2026,
  title   = {Safeguarding open-weight genomic foundation models through weight locking},
  author  = {Karatzikos, Aris and Vasilopoulou, Aggeliki and Chan, Candace SY and
             Mouratidis, Ioannis and Georgakopoulos-Soares, Ilias},
  journal = {},
  year    = {2026},
  url     = {https://github.com/Georgakopoulos-Soares-lab/glm-locking}
}
```
