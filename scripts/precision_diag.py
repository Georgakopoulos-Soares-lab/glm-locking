"""bf16 vs fp32 precision diagnostic for SpecDef-locked Evo-1 checkpoint.

Loads the locked ckpt once, runs forward over the same val batch under:
    (a) bf16 autocast  (production deployment)
    (b) fp32 (no autocast)         (high-precision)

Reports per-layer, per-block losses to isolate where precision drift originates.
Run on a single GPU.
"""
from __future__ import annotations

import os
import sys
import argparse
import math
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import load_evo_model, maybe_load_locked_checkpoint


def make_batch(tokenizer, seqs, seq_len, device):
    ids = []
    for s in seqs:
        toks = tokenizer.tokenize(s.upper())
        if len(toks) < seq_len + 1:
            toks = toks + [0] * (seq_len + 1 - len(toks))
        toks = toks[: seq_len + 1]
        ids.append(toks)
    arr = torch.tensor(ids, dtype=torch.long, device=device)
    return arr[:, :-1], arr[:, 1:]


def make_random_batches(tokenizer, seqs, batch_size, seq_len, n_batches, device):
    import random
    random.seed(42)
    batches = []
    for _ in range(n_batches):
        chunk_seqs = []
        for _ in range(batch_size):
            s = random.choice(seqs)
            if len(s) <= seq_len:
                chunk = s
            else:
                start = random.randint(0, len(s) - seq_len)
                chunk = s[start:start + seq_len]
            ids = tokenizer.tokenize(chunk)
            chunk_seqs.append(ids)
        batches.append(torch.tensor(chunk_seqs, dtype=torch.long, device=device))
    return batches


def eval_loss_avg(model, batches, dtype):
    """Match finetune.py evaluate(): average loss across n batches."""
    if dtype == torch.float32:
        ctx = torch.amp.autocast(device_type="cuda", enabled=False)
    else:
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    losses = []
    with torch.no_grad():
        for batch in batches:
            with ctx:
                out = model(batch)
                logits = out[0] if isinstance(out, tuple) else out.logits
                # finetune's causal_lm_loss: shift logits/targets, full xent
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = batch[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                )
            losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/lock_specdef_paper_all/model_specdef.pt")
    ap.add_argument("--fasta", default="data/attack_train.fasta")
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"Loading Evo model...")
    model, tokenizer = load_evo_model("evo-1-8k-base", args.device)
    model.eval()

    print(f"Loading locked checkpoint: {args.ckpt}")
    maybe_load_locked_checkpoint(model, args.ckpt)
    model.eval()

    # Read sequences (matching src.utils.load_sequences cleaning)
    from src.utils import load_sequences
    seqs = load_sequences(args.fasta, min_seq_len=args.seq_len)
    print(f"Loaded {len(seqs)} sequences from {args.fasta}")

    batches = make_random_batches(
        tokenizer, seqs, args.batch_size, args.seq_len, args.n_batches, args.device
    )
    print(f"Built {len(batches)} batches of {args.batch_size}×{args.seq_len}")

    # Inspect SpecDefLinear comp dtypes
    from scripts.lock_specdef import SpecDefLinear
    comp_dtypes = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SpecDefLinear):
            d = str(mod.comp.weight.dtype)
            comp_dtypes[d] = comp_dtypes.get(d, 0) + 1
    print(f"SpecDefLinear comp dtypes: {comp_dtypes}")

    # ---- Variant A: bf16 amp + SpecDefLinear (C in fp32, fp32 inner mat) ----
    loss_A = eval_loss_avg(model, batches, torch.bfloat16)
    print(f"[A] bf16 amp + SpecDefLinear      : {loss_A:.4f}  ppl={math.exp(loss_A):.2e}")

    # ---- Variant B: fp32 throughout ----
    loss_B = eval_loss_avg(model, batches, torch.float32)
    print(f"[B] fp32 (no amp) + SpecDefLinear : {loss_B:.4f}  ppl={math.exp(loss_B):.2e}")

    # ---- Variant C: bf16 amp, comp cast to bf16 (production FT freeze_comp=True) ----
    orig = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SpecDefLinear):
            orig[name] = mod.comp.weight.data.clone()
            mod.comp.weight.data = mod.comp.weight.data.to(torch.bfloat16)
    loss_C = eval_loss_avg(model, batches, torch.bfloat16)
    print(f"[C] bf16 amp + comp→bf16          : {loss_C:.4f}  ppl={math.exp(loss_C):.2e}")
    for name, mod in model.named_modules():
        if isinstance(mod, SpecDefLinear):
            mod.comp.weight.data = orig[name].to(mod.comp.weight.device)

    # ---- Variant D: specdef_fused_eval (bf16 fused linear) — exactly what finetune.py step-0 eval does ----
    from src.utils import specdef_fused_eval
    with specdef_fused_eval(model):
        loss_D = eval_loss_avg(model, batches, torch.bfloat16)
    print(f"[D] (C·W̃).bf16 fused             : {loss_D:.4f}  ppl={math.exp(loss_D):.2e}")

    # ---- Variant E: tiny perturbation to W̃ inflated weights ----
    # Simulate one optimizer step at lr=1e-6 with grad_clip=1.0:
    # update magnitude per param ~ lr * normalized_grad ~ 1e-6 * O(1)
    # Total l2 ~= sqrt(n_params) * 1e-6 ~= 0.084 for n=7e9 — matches FT log "l2=0.05"
    perturbations = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    pert_results = {}
    snapshot = {}
    for n, p in model.named_parameters():
        if ".comp.weight" in n or ".linear.weight" in n:  # SpecDef-related
            snapshot[n] = p.data.clone()
    print()
    print("Perturbation sensitivity test (Gaussian noise on SpecDef params):")
    g = torch.Generator(device=args.device).manual_seed(0)
    for sigma in perturbations:
        # restore + add noise
        for n, p in model.named_parameters():
            if n in snapshot:
                p.data = snapshot[n].clone()
                p.data.add_(torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype) * sigma)
        with specdef_fused_eval(model):
            l = eval_loss_avg(model, batches[:4], torch.bfloat16)
        pert_results[sigma] = l
        print(f"  sigma={sigma:.0e}  loss={l:.4f}  ppl={math.exp(l):.2e}")
    # restore
    for n, p in model.named_parameters():
        if n in snapshot:
            p.data = snapshot[n].clone()

    print()
    print("Summary (mean over %d random %d-tok chunks):" % (args.n_batches, args.seq_len))
    print(f"  [A] bf16 amp + SpecDefLinear (C in fp32)  : {loss_A:.4f}")
    print(f"  [B] fp32 throughout                        : {loss_B:.4f}")
    print(f"  [C] bf16 amp + comp→bf16 (freeze_comp)    : {loss_C:.4f}")
    print(f"  [D] (C·W̃).bf16 fused (FT step-0 eval)     : {loss_D:.4f}")
    print(f"  Pretrained val_loss baseline (unlocked)    : ~1.39")
    print(f"  Perturbation results (after Gaussian σ on W̃ and C):")
    for sigma, l in pert_results.items():
        print(f"    σ={sigma:.0e}  → loss={l:.4f}")

    pretrained_baseline = "expected ~1.39 (val_loss of unlocked)"
    print(f"  Pretrained baseline      : {pretrained_baseline}")

    # cleanup of dangling text moved above


if __name__ == "__main__":
    main()
