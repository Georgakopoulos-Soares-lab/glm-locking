#!/bin/bash
# FT sweep queue - GPU 6
set -u
cd /home/nvidia/glm-locking
export HF_HOME=/data/huggingface_cache
PY=/home/nvidia/miniconda3/envs/evo/bin/python3
LOG=results/hvue_finetune/sweep_gpu6.log
: > "$LOG"

run() {
  local ckpt="$1" path="$2" task="$3" lr="$4"
  echo "=== [$(date +%H:%M:%S)] $ckpt / $task (lr=$lr) ===" | tee -a "$LOG"
  if [ -z "$path" ]; then
    CUDA_VISIBLE_DEVICES=6 $PY scripts/hvue_finetune.py --ckpt "$ckpt" --task "$task" --lr "$lr" >> "$LOG" 2>&1
  else
    CUDA_VISIBLE_DEVICES=6 $PY scripts/hvue_finetune.py --ckpt "$ckpt" --ckpt_path "$path" --task "$task" --lr "$lr" >> "$LOG" 2>&1
  fi
  echo "--- done $ckpt / $task rc=$? ---" | tee -a "$LOG"
}

# pretrained (remaining 2 tasks)
run pretrained "" Pathogenecity 1e-5
run pretrained "" Transmissibility 1e-5
# M_a300k (remaining 2 tasks)
run M_a300k results/ft_locked_a300k_lr1e5_25k_locked/model_finetuned.pt Pathogenecity 3e-6
run M_a300k results/ft_locked_a300k_lr1e5_25k_locked/model_finetuned.pt Transmissibility 3e-6
# K_a100k (all 3)
run K_a100k results/ft_locked_a100k_lr1e5_25k_locked/model_finetuned.pt Host_Tropism 3e-6
run K_a100k results/ft_locked_a100k_lr1e5_25k_locked/model_finetuned.pt Pathogenecity 3e-6
run K_a100k results/ft_locked_a100k_lr1e5_25k_locked/model_finetuned.pt Transmissibility 3e-6
# unlocked (Host_Tropism only here; others on gpu7)
run unlocked_ft results/ft_unlocked_full910_25k_unlocked/model_finetuned.pt Host_Tropism 1e-5

echo "=== GPU6 QUEUE COMPLETE [$(date +%H:%M:%S)] ===" | tee -a "$LOG"
