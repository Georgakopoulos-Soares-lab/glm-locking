"""ViroBench PPL evaluation: sweep across ALL lock strengths + controls.

Lock-only models (no attack): measure pure SpecDef PPL drift.
Attacked models (locked + finetuned): measure post-attack PPL.
Uses specdef_fused_eval for fair comparison (fuses C@W̃ → bf16).

Usage:
  CUDA_VISIBLE_DEVICES=6 python experiments/exp3_virobench/virobench_ppl.py
"""
from __future__ import annotations
import os, sys, argparse, math, time, random
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from src.utils import load_evo_model, maybe_load_locked_checkpoint, specdef_fused_eval


# ── Config ──────────────────────────────────────────────────────────────────
MODELS = [
    # Controls
    ("pretrained",          None,                                                         "Pretrained"),
    ("unlocked_ft_910",     "results/ft_unlocked_full910_25k_unlocked/model_finetuned.pt", "Unlocked-FT (910)"),
    # Lock-only (no attack) — pure SpecDef PPL drift
    ("lock_a10k",           "results/lock_alpha10k/model_specdef.pt",                      "Locked α=10⁴"),
    ("lock_a30k",           "results/lock_alpha30k/model_specdef.pt",                      "Locked α=3×10⁴"),
    ("lock_a100k",          "results/lock_alpha100k/model_specdef.pt",                     "Locked α=10⁵"),
    ("lock_a300k",          "results/lock_alpha300k/model_specdef.pt",                     "Locked α=3×10⁵"),
    # Attacked (locked + finetuned)
    ("M_a300k",             "results/ft_locked_a300k_lr1e5_25k_locked/model_finetuned.pt",  "M (α=3×10⁵, attacked)"),
]

SEQ_LEN = 1024
BATCH_SIZE = 1
N_BATCHES = 256
MAX_SEQ_LEN = 1024
OUT_CSV = "experiments/exp3_virobench/virobench_ppl_all_locks.csv"
SEED = 42


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_virobench_sequences():
    """Load ViroBench test split sequences."""
    from datasets import load_dataset
    ds = load_dataset("YDXX/ViroBench")
    seqs = [row["sequences"][0] for row in ds["test"]]
    # Filter to sequences >= SEQ_LEN
    seqs = [s for s in seqs if len(s) >= SEQ_LEN]
    print(f"Loaded {len(seqs)} test sequences (≥{SEQ_LEN}nt)")
    return seqs


def make_batches(tokenizer, seqs, batch_size, seq_len, n_batches, device, seed):
    """Create random batches of tokenized chunks."""
    random.seed(seed)
    out = []
    for i in range(n_batches):
        rows = []
        for _ in range(batch_size):
            s = random.choice(seqs)
            if len(s) <= seq_len:
                chunk = s
            else:
                start = random.randint(0, len(s) - seq_len)
                chunk = s[start:start + seq_len]
            ids = tokenizer.tokenize(chunk)
            rows.append(ids)
        out.append(torch.tensor(rows, dtype=torch.long, device=device))
    return out


def ppl_eval(model, batches, dtype):
    """Compute average loss and perplexity."""
    losses = []
    with torch.no_grad():
        for batch in batches:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                out = model(batch)
                logits = out[0] if isinstance(out, tuple) else out.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = batch[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                )
            losses.append(loss.item())
    avg = sum(losses) / len(losses)
    return avg, math.exp(avg)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    device = "cuda:0"

    # 1. Load sequences and tokenizer
    seqs = load_virobench_sequences()
    from evo import Evo
    tokenizer = Evo("evo-1-8k-base").tokenizer

    import csv as _csv
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # 2. Evaluate each model
    rows = []
    for ckpt_name, ckpt_path, display_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {display_name}")
        print(f"{'='*60}")

        # Load model
        model, _ = load_evo_model("evo-1-8k-base", device)
        maybe_load_locked_checkpoint(model, ckpt_path)
        model.eval()

        # Create batches (same seed per model for fair comparison)
        batches = make_batches(tokenizer, seqs, BATCH_SIZE, SEQ_LEN, N_BATCHES, device, SEED)

        dtype = torch.bfloat16

        # For locked models: use fused eval for fair PPL comparison
        with specdef_fused_eval(model):
            t0 = time.time()
            loss, ppl = ppl_eval(model, batches, dtype)
            elapsed = time.time() - t0

        print(f"  Loss: {loss:.4f}  PPL: {ppl:.2f}  ({elapsed:.1f}s)")

        rows.append({
            "name": ckpt_name,
            "display": display_name,
            "loss": f"{loss:.6f}",
            "ppl": f"{ppl:.4f}",
            "n_batches": N_BATCHES,
            "seq_len": SEQ_LEN,
        })

        del model
        torch.cuda.empty_cache()

    # 3. Save results
    keys = ["name", "display", "loss", "ppl", "n_batches", "seq_len"]
    with open(OUT_CSV, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {OUT_CSV}")

    # 4. Print comparison
    print(f"\n{'='*60}")
    print("PPL Comparison on ViroBench Test")
    print(f"{'='*60}")
    for r in rows:
        print(f"  {r['display']:20s}  PPL={r['ppl']}")


if __name__ == "__main__":
    main()
