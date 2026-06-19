#!/bin/bash
# FT sweep queue - GPU 7
set -u
cd /home/nvidia/glm-locking
export HF_HOME=/data/huggingface_cache
PY=/home/nvidia/miniconda3/envs/evo/bin/python3
LOG=results/hvue_finetune/sweep_gpu7.log
: > "$LOG"

run() {
  local ckpt="$1" path="$2" task="$3" lr="$4"
  echo "=== [$(date +%H:%M:%S)] $ckpt / $task (lr=$lr) ===" | tee -a "$LOG"
  if [ -z "$path" ]; then
    CUDA_VISIBLE_DEVICES=7 $PY scripts/hvue_finetune.py --ckpt "$ckpt" --task "$task" --lr "$lr" >> "$LOG" 2>&1
  else
    CUDA_VISIBLE_DEVICES=7 $PY scripts/hvue_finetune.py --ckpt "$ckpt" --ckpt_path "$path" --task "$task" --lr "$lr" >> "$LOG" 2>&1
  fi
  echo "--- done $ckpt / $task rc=$? ---" | tee -a "$LOG"
}

# unlocked (remaining 2 tasks)
run unlocked_ft results/ft_unlocked_full910_25k_unlocked/model_finetuned.pt Pathogenecity 1e-5
run unlocked_ft results/ft_unlocked_full910_25k_unlocked/model_finetuned.pt Transmissibility 1e-5
# SVD best 1 (a30k_k1) all 3
run SVD_a30k_k1 results/ft_theorem8_a30k_k1_25k_locked/model_finetuned.pt Host_Tropism 3e-6
run SVD_a30k_k1 results/ft_theorem8_a30k_k1_25k_locked/model_finetuned.pt Pathogenecity 3e-6
run SVD_a30k_k1 results/ft_theorem8_a30k_k1_25k_locked/model_finetuned.pt Transmissibility 3e-6
# SVD best 2 (a10k_k3) all 3
run SVD_a10k_k3 results/ft_theorem8_a10k_k3_25k_locked/model_finetuned.pt Host_Tropism 3e-6
run SVD_a10k_k3 results/ft_theorem8_a10k_k3_25k_locked/model_finetuned.pt Pathogenecity 3e-6
run SVD_a10k_k3 results/ft_theorem8_a10k_k3_25k_locked/model_finetuned.pt Transmissibility 3e-6

echo "=== GPU7 QUEUE COMPLETE [$(date +%H:%M:%S)] ===" | tee -a "$LOG"
