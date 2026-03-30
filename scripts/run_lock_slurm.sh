#!/bin/bash
# SpecDef locking — SLURM launcher for TACC Stampede3 / other h100 partitions.
#
# Single GPU:
#   sbatch --gres=gpu:1 --ntasks-per-node=1 scripts/run_lock_slurm.sh
#
# 4-GPU single node (DDP via torchrun):
#   sbatch --gres=gpu:4 --ntasks-per-node=1 scripts/run_lock_slurm.sh
#
# GPU count is inferred automatically via SLURM_GPUS_ON_NODE.

#SBATCH -J evo_lock
#SBATCH -o logs/evo_lock.%j.out
#SBATCH -e logs/evo_lock.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH --mem=40G
#SBATCH -t 2:00:00
#SBATCH -A BCS25073

module purge
unset LD_PRELOAD

CONDA_ROOT="/work/10906/arisk/conda"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate evo
elif [ -f "$CONDA_ROOT/bin/activate" ]; then
    source "$CONDA_ROOT/bin/activate" evo
else
    echo "[ERROR] Cannot find conda at $CONDA_ROOT" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="/work/10906/arisk/ls6/evo-locking:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /work/10906/arisk/ls6/evo-locking
mkdir -p logs

# Determine GPU count (defaults to 1 if not set by SLURM)
NGPUS="${SLURM_GPUS_ON_NODE:-1}"
echo "Launching SpecDef locking on ${NGPUS} GPU(s)"

if [ "$NGPUS" -gt 1 ]; then
    # Multi-GPU: torchrun spawns one process per GPU, sets LOCAL_RANK automatically
    torchrun \
        --standalone \
        --nproc_per_node="${NGPUS}" \
        scripts/lock.py
else
    # Single GPU: plain python (LOCAL_RANK not set → DDP disabled in lock.py)
    python scripts/lock.py
fi
