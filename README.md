# Evo Locking

This repository contains a clean workspace for pilot experiments on weight locking in Evo on a viral pilot dataset.

## Scripts

- `Evo_viral_ft_utils.py`  
  Shared utilities for FASTA loading, train/validation splitting, random window sampling, evaluation, optimizer setup, parameter counting, and reporting.

- `Evo_viral_eval_pretrained.py`  
  Evaluates the pretrained Evo model on the viral pilot dataset and reports both train and validation metrics.

- `Evo_viral_ft_blocks_unlocked.py`  
  Runs broader block-restricted fine-tuning on blocks 0-7 with unlocked initialization.

- `Evo_viral_ft_blocks_locked.py`  
  Runs broader block-restricted fine-tuning on blocks 0-7 starting from the locked checkpoint.

## Result folders

The `results/` directory contains the following runs:

- `eval_pretrained_pilot758_len128_clean/`
- `ft_blocks_unlocked_pilot758_len128_clean/`
- `ft_blocks_locked_v7_pilot758_len128_clean/`
- `ft_blocks_unlocked_pilot758_len128_clean_2k/`
- `ft_blocks_locked_v7_pilot758_len128_clean_2k/`

## Dataset / setup

These runs use the viral pilot FASTA dataset:

- 758 usable sequences total
- 682 training sequences
- 76 validation sequences
- approximate sequence length range: 8.4 kb to 30.1 kb
- sequence window length: 128
- base model: `evo-1-8k-base`

## Metrics reported

The clean scripts report:

- train loss / perplexity / next-token accuracy
- validation loss / perplexity / next-token accuracy
- total parameters
- trainable parameter count and fraction
- targeted parameter count and fraction
- training steps
- tokens seen
- approximate train-data coverage ratio
