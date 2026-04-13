"""Evaluate attack-task perplexity on a clean, fused basis.

Re-creates the exact val split used during training (seed=42, 90/10),
then measures causal-LM loss on each checkpoint using specdef_fused_eval
so that locked and unlocked checkpoints are evaluated under identical conditions.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n evo --no-capture-output \
        python -u scripts/eval_attack_ppl.py
"""

import os, sys, json, math, random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    set_seed, get_amp_settings, clean_dna,
    load_evo_model, maybe_load_locked_checkpoint,
    causal_lm_loss, specdef_fused_eval,
    split_sequences,
)

DEVICE  = "cuda:0"
SEQ_LEN = 1024
SEED    = 42
OUT_DIR = "results/eval_attack_ppl"

CHECKPOINTS = [
    ("pretrained",             None),
    ("unlocked_lr1e5_best",    "results/ft_paper_lr1e5_unlocked/model_best.pt"),
    ("locked_paper_lr1e5_best","results/ft_paper_lr1e5_locked/model_best.pt"),
    ("unlocked_topk5_best",    "results/ft_gentler_topk5_unlocked/model_best.pt"),
    ("locked_topk5_best",      "results/ft_gentler_topk5_locked/model_best.pt"),
]


def parse_fasta(path, min_len=100):
    records = []
    header, chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and chunks:
                    seq = clean_dna("".join(chunks))
                    if len(seq) >= min_len:
                        records.append(seq)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None and chunks:
        seq = clean_dna("".join(chunks))
        if len(seq) >= min_len:
            records.append(seq)
    return records


def compute_mean_loss(model, tokenizer, seqs, device, seq_len, amp_dtype):
    """Compute mean causal-LM loss (nats) over sequences, using centre-crop."""
    losses = []
    for seq in seqs:
        ids = list(tokenizer.tokenize(seq[:seq_len * 2]))
        if len(ids) > seq_len:
            start = (len(ids) - seq_len) // 2
            ids = ids[start:start + seq_len]
        if len(ids) < 2:
            continue
        t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = model(t)
            loss = causal_lm_loss(logits, t)
        losses.append(loss.item())
    return float(np.mean(losses)), float(np.std(losses)), len(losses)


def main():
    amp_dtype, _ = get_amp_settings()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Replicate exact val split from training (seed=42, train_fraction=0.9)
    set_seed(SEED)
    all_seqs = parse_fasta("data/attack.fasta")
    _, val_seqs = split_sequences(all_seqs, train_fraction=0.9)
    print(f"Attack val set: {len(val_seqs)} sequences (same split as training)", flush=True)

    out_path = os.path.join(OUT_DIR, "results.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    for ckpt_name, ckpt_path in CHECKPOINTS:
        print(f"\n{'='*60}", flush=True)
        print(f"Checkpoint: {ckpt_name}", flush=True)

        model, tokenizer = load_evo_model("evo-1-8k-base", DEVICE)
        maybe_load_locked_checkpoint(model, ckpt_path)
        model.eval()

        with specdef_fused_eval(model):
            mean_loss, std_loss, n = compute_mean_loss(
                model, tokenizer, val_seqs, DEVICE, SEQ_LEN, amp_dtype
            )
        ppl = math.exp(mean_loss)
        print(f"  attack val_loss = {mean_loss:.4f} ± {std_loss:.4f}  PPL = {ppl:.4f}  (n={n})",
              flush=True)

        record = {
            "checkpoint": ckpt_name,
            "val_loss": mean_loss,
            "val_loss_std": std_loss,
            "val_ppl": ppl,
            "n": n,
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY — Attack-task val loss (fused eval)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Checkpoint':35s}  {'val_loss':>9s}  {'val_PPL':>8s}  {'Δ from pretrained':>18s}", flush=True)
    print("-" * 75, flush=True)
    baseline = None
    rows = []
    with open(out_path) as f:
        for line in f:
            rows.append(json.loads(line))
    baseline = rows[0]["val_loss"]
    for r in rows:
        delta = r["val_loss"] - baseline
        sign = "+" if delta >= 0 else ""
        print(f"{r['checkpoint']:35s}  {r['val_loss']:>9.4f}  {r['val_ppl']:>8.4f}  {sign}{delta:>+.4f}", flush=True)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
