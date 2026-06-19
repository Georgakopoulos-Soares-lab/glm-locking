#!/usr/bin/env bash
# Parallel LoRA fine-tuning across 8 GPUs.
#
# Assignment (12 jobs, 8 GPUs — GPUs 0-3 chain 2 jobs each):
#   GPU 0: Host_Tropism    × pretrained    THEN Transmissibility × pretrained
#   GPU 1: Host_Tropism    × locked_no_ft  THEN Transmissibility × locked_no_ft
#   GPU 2: Host_Tropism    × unlocked_ft   THEN Transmissibility × unlocked_ft
#   GPU 3: Host_Tropism    × M_a300k       THEN Transmissibility × M_a300k
#   GPU 4: Pathogenecity   × pretrained
#   GPU 5: Pathogenecity   × locked_no_ft
#   GPU 6: Pathogenecity   × unlocked_ft
#   GPU 7: Pathogenecity   × M_a300k
#
# Each job writes to its own CSV: results/hvue_lora_{task}_{ckpt}.csv
# Merge all CSVs afterwards: python scripts/merge_lora_results.py
#
# Time estimate: ~130 min per LR run (early stop), 3 LRs = ~6.5h per (task,ckpt).
# Single-job GPUs (4-7): ~6.5h. Chained GPUs (0-3): ~13h.
# Total wall time: ~13h.

set -e
REPO=/home/nvidia/glm-locking
PY=/home/nvidia/miniconda3/envs/evo/bin/python3
SCRIPT=$REPO/scripts/hvue_lora_finetune.py
OUTDIR=$REPO/results/hvue_lora
mkdir -p $OUTDIR $REPO/logs/hvue_lora

ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# Seed the already-done result (pretrained × Host_Tropism × LR=1e-4)
$PY -c "
import csv, os
out='$OUTDIR/Host_Tropism_pretrained.csv'
row = dict(ckpt='pretrained', task='Host_Tropism', lr=1e-4,
           best_val_auroc=0.9359, best_step=1500,
           n_lora_params=34480128, n_lora_layers=154,
           is_locked=False, kmer_auroc=0.9030, residual=0.0329)
with open(out,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(row.keys()))
    w.writeheader(); w.writerow(row)
print('seeded', out)
"

run_job() {
    local gpu=$1
    local task=$2
    local ckpt=$3
    local out="$OUTDIR/${task}_${ckpt}.csv"
    local log="$REPO/logs/hvue_lora/${task}_${ckpt}.log"
    echo "[GPU $gpu] launching: $task × $ckpt  →  $out"
    env $ENV CUDA_VISIBLE_DEVICES=$gpu \
        nohup $PY $SCRIPT \
            --tasks "$task" \
            --ckpts "$ckpt" \
            --lrs 1e-4 5e-5 1e-5 \
            --resume \
            --out "$out" \
        > "$log" 2>&1 &
    echo $! > "$REPO/logs/hvue_lora/${task}_${ckpt}.pid"
    echo "    PID $!"
}

chain_job() {
    local gpu=$1
    local task1=$2; local ckpt1=$3
    local task2=$4; local ckpt2=$5
    local out1="$OUTDIR/${task1}_${ckpt1}.csv"
    local out2="$OUTDIR/${task2}_${ckpt2}.csv"
    local log="$REPO/logs/hvue_lora/${task1}_${ckpt1}__then__${task2}_${ckpt2}.log"
    echo "[GPU $gpu] chaining: $task1×$ckpt1  THEN  $task2×$ckpt2"
    env $ENV CUDA_VISIBLE_DEVICES=$gpu bash -c "
        $PY $SCRIPT --tasks '$task1' --ckpts '$ckpt1' --lrs 1e-4 5e-5 1e-5 --resume --out '$out1' &&
        $PY $SCRIPT --tasks '$task2' --ckpts '$ckpt2' --lrs 1e-4 5e-5 1e-5 --resume --out '$out2'
    " > "$log" 2>&1 &
    echo "    PID $!"
}

# Chained pairs on GPUs 0-3
chain_job 0 Host_Tropism pretrained      Transmissibility pretrained
chain_job 1 Host_Tropism locked_no_ft    Transmissibility locked_no_ft
chain_job 2 Host_Tropism unlocked_ft     Transmissibility unlocked_ft
chain_job 3 Host_Tropism M_a300k         Transmissibility M_a300k

# Single jobs on GPUs 4-7
run_job 4 Pathogenecity pretrained
run_job 5 Pathogenecity locked_no_ft
run_job 6 Pathogenecity unlocked_ft
run_job 7 Pathogenecity M_a300k

echo ""
echo "All 8 GPU jobs launched."
echo "Monitor: tail -f /home/nvidia/glm-locking/logs/hvue_lora/*.log"
echo "Merge:   python scripts/merge_lora_results.py"
