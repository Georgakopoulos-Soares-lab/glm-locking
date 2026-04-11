"""Biological evaluation: compare pretrained, unlocked-finetuned, and locked-finetuned
checkpoints on the attack virus sequences.

For each checkpoint × dataset pair, computes per-sequence perplexity using
sliding windows of seq_len with stride seq_len//2.

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -n evo --no-capture-output \
        python -u scripts/bio_eval.py
"""

import os, sys, json, math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    set_seed, get_amp_settings, load_sequences,
    load_evo_model, maybe_load_locked_checkpoint,
    causal_lm_loss, specdef_fused_eval,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE      = "cuda:0"
SEQ_LEN     = 1024
STRIDE      = 512
BATCH_SIZE  = 2
MIN_LEN     = 512
SEED        = 42

CHECKPOINTS = [
    ("pretrained",        None),
    ("unlocked_ft",       "results/ft_paper_lr1e5_unlocked/model_best.pt"),
    ("locked_ft_lr1e5",   "results/ft_paper_lr1e5_locked/model_best.pt"),
    ("locked_ft_lr3e5",   "results/ft_paper_lr3e5_locked/model_best.pt"),
]

# Use the attack FASTA (virus sequences) — the same data the fine-tuning used
DATASETS = [
    ("attack_viruses", "data/attack.fasta"),
    ("retain_bacteria", "data/retain.fasta"),
]

OUT_PATH = "results/bio_eval_results.jsonl"

# ---------------------------------------------------------------------------
# Per-sequence perplexity via sliding windows
# ---------------------------------------------------------------------------
def seq_perplexity(model, tokenizer, seq: str, device: str,
                   seq_len: int, stride: int, amp_dtype) -> float:
    """Compute perplexity of a single sequence using overlapping windows."""
    # tokenizer.tokenize returns a 1-D array of int token ids (same as build_batch)
    ids = list(tokenizer.tokenize(seq))
    total_loss = 0.0
    total_tokens = 0
    n = len(ids)

    starts = list(range(0, max(1, n - seq_len), stride))
    if not starts:
        starts = [0]
    # Always include last window
    if starts[-1] + seq_len < n:
        starts.append(max(0, n - seq_len))

    with torch.no_grad():
        for s in starts:
            chunk = ids[s: s + seq_len]
            if len(chunk) < 2:
                continue
            t = torch.tensor([chunk], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, _ = model(t)
                loss = causal_lm_loss(logits, t)
            n_tok = t.shape[1] - 1
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

    if total_tokens == 0:
        return float('nan')
    return math.exp(total_loss / total_tokens)


def eval_dataset(model, tokenizer, fasta_path: str, device: str,
                 seq_len: int, stride: int, amp_dtype,
                 max_seqs: int = 200) -> dict:
    """Evaluate perplexity on all sequences in a FASTA file."""
    seqs = load_sequences(fasta_path, min_seq_len=MIN_LEN)
    if max_seqs and len(seqs) > max_seqs:
        import random
        rng = random.Random(SEED)
        seqs = rng.sample(seqs, max_seqs)

    ppls = []
    for i, seq in enumerate(seqs):
        ppl = seq_perplexity(model, tokenizer, seq, device, seq_len, stride, amp_dtype)
        ppls.append(ppl)
        if (i + 1) % 20 == 0:
            valid = [p for p in ppls if not math.isnan(p)]
            print(f"    {i+1}/{len(seqs)}  running mean PPL={sum(valid)/len(valid):.3f}", flush=True)

    valid = [p for p in ppls if not math.isnan(p)]
    return {
        "n_seqs": len(seqs),
        "n_valid": len(valid),
        "mean_ppl": sum(valid) / len(valid) if valid else float('nan'),
        "median_ppl": sorted(valid)[len(valid)//2] if valid else float('nan'),
        "min_ppl":  min(valid) if valid else float('nan'),
        "max_ppl":  max(valid) if valid else float('nan'),
        "per_seq_ppl": valid,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    amp_dtype, _ = get_amp_settings()
    os.makedirs("results", exist_ok=True)

    # Remove old results
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    for ckpt_name, ckpt_path in CHECKPOINTS:
        print(f"\n{'='*60}", flush=True)
        print(f"Loading checkpoint: {ckpt_name}", flush=True)
        model, tokenizer = load_evo_model("evo-1-8k-base", DEVICE)
        maybe_load_locked_checkpoint(model, ckpt_path)
        model.eval()

        for ds_name, ds_path in DATASETS:
            print(f"\n  Dataset: {ds_name}", flush=True)
            with specdef_fused_eval(model):
                result = eval_dataset(model, tokenizer, ds_path,
                                      DEVICE, SEQ_LEN, STRIDE, amp_dtype,
                                      max_seqs=200)

            record = {
                "checkpoint": ckpt_name,
                "dataset": ds_name,
                **{k: v for k, v in result.items() if k != "per_seq_ppl"},
            }
            print(f"  -> mean_ppl={record['mean_ppl']:.4f}  "
                  f"median_ppl={record['median_ppl']:.4f}  "
                  f"n={record['n_valid']}", flush=True)

            with open(OUT_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")

        # Free GPU memory before next checkpoint
        del model
        torch.cuda.empty_cache()

    # --- Summary table ---
    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print(f"{'Checkpoint':25s}  {'Dataset':20s}  {'Mean PPL':>9}  {'Median PPL':>11}", flush=True)
    print("-"*72, flush=True)
    with open(OUT_PATH) as f:
        for line in f:
            r = json.loads(line)
            print(f"{r['checkpoint']:25s}  {r['dataset']:20s}  "
                  f"{r['mean_ppl']:>9.4f}  {r['median_ppl']:>11.4f}", flush=True)


if __name__ == "__main__":
    main()
