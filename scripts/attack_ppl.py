"""Compute attack_heldout PPL for many checkpoints.

Same eval procedure as finetune.py: bf16 amp, specdef_fused_eval,
random 1024-tok chunks, average loss over n_batches.
"""
from __future__ import annotations
import os, sys, argparse, math, json, time
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import load_evo_model, maybe_load_locked_checkpoint, load_sequences, specdef_fused_eval
import random


def make_batches(tokenizer, seqs, batch_size, seq_len, n_batches, device, seed):
    random.seed(seed)
    out = []
    for _ in range(n_batches):
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


# Default checkpoint registry, using canonical paths produced by configs/lock/*
# and configs/finetune/*. Override via --ckpts_csv path/to/registry.csv (cols:
# name,ckpt) or by editing this list. None or empty path = pretrained Evo.
CKPTS = [
    # name, ckpt_path
    ("pretrained",                  None),
    # unlocked baseline (regular FT, no SpecDef)
    ("ft_unlocked_25k_best",        "results/ft_unlocked_25k/model_best.pt"),
    ("ft_unlocked_25k_final",       "results/ft_unlocked_25k/model_finetuned.pt"),
    # loaded-only locked at various alpha (no FT, sanity check)
    ("locked_a10k",                 "results/lock_alpha10k/model_specdef.pt"),
    ("locked_a30k",                 "results/lock_alpha30k/model_specdef.pt"),
    ("locked_a100k",                "results/lock_alpha100k/model_specdef.pt"),
    ("locked_a1M",                  "results/lock_alpha1M/model_specdef.pt"),
    # ft locked at stable LR (lr ~ 1/alpha)
    ("ft_locked_a10k_lr1e6",        "results/ft_locked_a10k_lr1e6/model_best.pt"),
    ("ft_locked_a30k_lr1e6",        "results/ft_locked_a30k_lr1e6/model_best.pt"),
    ("ft_locked_a100k_lr1e6",       "results/ft_locked_a100k_lr1e6/model_best.pt"),
    ("ft_locked_a1M_lr1e7",         "results/ft_locked_a1M_lr1e7/model_best.pt"),
    # ft locked at destabilized LRs (instability evidence)
    ("ft_locked_a10k_lr1e4",        "results/ft_locked_a10k_lr1e4/model_best.pt"),
    ("ft_locked_a10k_lr3e5",        "results/ft_locked_a10k_lr3e5/model_best.pt"),
    ("ft_locked_a10k_lr3e6",        "results/ft_locked_a10k_lr3e6/model_best.pt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="data/attack_heldout.fasta")
    ap.add_argument("--n_batches", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/attack_heldout_ppl.csv")
    ap.add_argument("--ckpts_csv", default=None,
                    help="Optional CSV with cols name,ckpt to override CKPTS")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    if args.ckpts_csv:
        import csv as _csv
        ckpts = []
        with open(args.ckpts_csv) as f:
            for r in _csv.DictReader(f):
                p = r.get("ckpt") or None
                if p in ("", "pretrained", "None"):
                    p = None
                ckpts.append((r["name"], p))
    else:
        ckpts = CKPTS

    seqs = load_sequences(args.fasta, min_seq_len=args.seq_len)
    print(f"Loaded {len(seqs)} sequences from {args.fasta}")

    import csv, gc
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Append mode: read existing rows so we can resume
    existing = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            for r in csv.DictReader(f):
                existing[r["name"]] = r
        print(f"Found {len(existing)} existing rows in {args.out}")
    rows = list(existing.values())

    def write():
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name","ckpt","val_loss","val_ppl"])
            w.writeheader()
            for r in rows: w.writerow(r)

    for name, ckpt in ckpts:
        if args.only and name not in args.only:
            continue
        if name in existing:
            print(f"  SKIP {name}: already done")
            continue
        if ckpt is not None and not os.path.exists(ckpt):
            print(f"  SKIP {name}: missing {ckpt}")
            continue
        print(f"\n=== {name} ===")
        t0 = time.time()
        model, tok = load_evo_model("evo-1-8k-base", args.device)
        if ckpt is not None:
            maybe_load_locked_checkpoint(model, ckpt)
        model.eval()
        batches = make_batches(tok, seqs, args.batch_size, args.seq_len,
                               args.n_batches, args.device, args.seed)
        with specdef_fused_eval(model):
            loss, ppl = ppl_eval(model, batches, torch.bfloat16)
        dt = time.time() - t0
        print(f"  val_loss={loss:.4f}  val_ppl={ppl:.4f}  ({dt:.1f}s)")
        rows.append(dict(name=name, ckpt=ckpt or "pretrained", val_loss=loss, val_ppl=ppl))
        write()
        # Aggressive cleanup to avoid OOM accumulation
        del model, tok, batches
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    print()
    print(f"{'name':40s}  {'val_loss':>8s}  {'val_ppl':>7s}")
    for r in rows:
        # rows loaded from CSV are strings; cast for formatting
        vl = float(r['val_loss']); vp = float(r['val_ppl'])
        print(f"{r['name']:40s}  {vl:8.4f}  {vp:7.3f}")


if __name__ == "__main__":
    main()
