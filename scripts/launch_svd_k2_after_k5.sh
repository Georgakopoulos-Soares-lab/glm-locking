#!/usr/bin/env bash
# Wait for svd_k5 (PID 4026694) to finish, then launch svd_k2 on GPU 1
set -euo pipefail
cd /home/nvidia/glm-locking
mkdir -p logs/lora_fair results/hvue_lora_shards

K5_PID=4026694
PY=/home/nvidia/miniconda3/envs/evo/bin/python3
SCRIPT=scripts/hvue_lora_finetune.py
TASKS="Host_Tropism Pathogenecity Transmissibility"

echo "[$(date)] Waiting for svd_k5 (PID=$K5_PID) to finish..."
while kill -0 "$K5_PID" 2>/dev/null; do
    sleep 120
done
echo "[$(date)] svd_k5 done. Launching svd_k2 on GPU 1..."

CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT \
    --tasks $TASKS \
    --ckpts svd_k2_a30k \
    --lrs 1e-4 5e-5 1e-5 \
    --lora_config matched \
    --out results/hvue_lora_shards/svd_k2.csv \
    --resume \
    >> logs/lora_fair/svd_k2.log 2>&1

echo "[$(date)] svd_k2 complete."
