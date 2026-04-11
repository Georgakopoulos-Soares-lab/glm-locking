#!/usr/bin/env bash
set -euo pipefail

# 1) Fresh conda env
conda create -n evo python=3.10 -y

# Initialize conda in non-interactive shells, then activate env
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Conda not initialized. Ensure conda is installed and available in PATH." >&2
  exit 1
fi
conda activate evo

# 2) Install the GPU build of PyTorch that matches CUDA 12.8
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0

# 3) FlashAttention built for your Torch/CUDA
conda install -c conda-forge flash-attn=2.7.4 -y

# 4) Evo model (depends on torch; do it AFTER torch)
pip install evo-model

# 5) Extra Python libs (pin nx to avoid the dataclass backend issue)
pip install --no-cache-dir "networkx==3.2.1" biopython pandas scipy matplotlib seaborn

echo "Environment 'evo' is ready."
