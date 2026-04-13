"""Fusion bypass: fuse C@W̃ → W and save as a plain model checkpoint.

This is the trivial open-weight attacker bypass:
  1. Load the SpecDef locked checkpoint (has inflated weights W̃ and compensation C)
  2. For each locked layer: compute W_recovered = C @ W̃  (exact pretrained weights)
  3. Save a plain state dict with standard nn.Linear weights

Since C·W̃ = W exactly (algebraic guarantee), the output is bit-identical to
the pretrained checkpoint. Any attacker with access to model_specdef.pt can
run this in ~10 seconds and fine-tune freely.

Usage:
    python scripts/fuse_specdef_checkpoint.py \
        --input  results/lock_specdef_paper_all/model_specdef.pt \
        --output results/model_fused_pretrained.pt

    # Verify PPL matches pretrained:
    python scripts/eval_locked_specdef_only.py  # (add fused to CHECKPOINTS)
"""
import argparse
import os
import re
import time

import torch


def fuse_specdef(input_path: str, output_path: str):
    print(f"Loading SpecDef checkpoint: {input_path}")
    t0 = time.time()
    state_dict = torch.load(input_path, map_location="cpu", weights_only=False)

    comp_keys = [k for k in state_dict if k.endswith(".comp.weight")]
    linear_keys = [k for k in state_dict if k.endswith(".linear.weight")]

    if not comp_keys:
        raise ValueError("No .comp.weight keys found — is this a SpecDef checkpoint?")

    print(f"Found {len(comp_keys)} locked layers to fuse")

    fused = {}
    n_replaced = 0

    for key, val in state_dict.items():
        # Fuse: blocks.X.PATTERN.linear.weight → blocks.X.PATTERN.weight
        m = re.match(r"^(blocks\.\d+\..+)\.linear\.weight$", key)
        if m:
            prefix = m.group(1)
            comp_key = f"{prefix}.comp.weight"
            if comp_key in state_dict:
                W_tilde = val.float()       # [m, n]
                C = state_dict[comp_key].float()  # [m, m]
                W_recovered = C @ W_tilde   # [m, n] — exact pretrained weight
                # Store as bfloat16 (matches Evo's native dtype)
                fused[f"{prefix}.weight"] = W_recovered.bfloat16()
                n_replaced += 1
                continue  # skip storing .linear.weight

        # Drop all .comp.weight keys (no longer needed)
        if key.endswith(".comp.weight"):
            continue

        # Drop .bias keys that belonged to SpecDefLinear wrappers
        # (bias is stored separately; the original nn.Linear had no bias for these layers)
        # Keep everything else as-is
        fused[key] = val

    print(f"Fused {n_replaced} layers in {time.time()-t0:.1f}s")
    print(f"State dict: {len(state_dict)} keys → {len(fused)} keys")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(fused, output_path)
    print(f"Saved fused checkpoint: {output_path}")
    return fused


def verify_ppl(fused_path: str, device: str = "cuda:0"):
    """Quick sanity check: load fused model and compute PPL on a few sequences."""
    import sys, math, random
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.utils import (set_seed, get_amp_settings, clean_dna,
                           load_evo_model, causal_lm_loss)

    set_seed(42)
    amp_dtype, _ = get_amp_settings()

    print(f"\nVerifying fused model PPL on GPU ({device})...")
    model, tokenizer = load_evo_model("evo-1-8k-base", device)

    # Load fused weights directly (plain state dict, no SpecDefLinear)
    fused_sd = torch.load(fused_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(fused_sd, strict=False)
    print(f"  Missing={len(missing)}, Unexpected={len(unexpected)}")

    model.eval()

    # Load a few sequences
    records = []
    with open("data/retain.fasta") as f:
        header, chunks = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header and chunks:
                    seq = clean_dna("".join(chunks))
                    if len(seq) >= 512:
                        records.append(seq)
                header = line[1:]; chunks = []
            else:
                chunks.append(line)
    sample = random.Random(42).sample(records, min(50, len(records)))

    ppls = []
    for seq in sample:
        ids = list(tokenizer.tokenize(seq[:2048]))
        if len(ids) > 1024:
            ids = ids[:1024]
        if len(ids) < 2:
            continue
        t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = model(t)
            loss = causal_lm_loss(logits, t)
        ppls.append(math.exp(loss.item()))

    import numpy as np
    print(f"  Fused model PPL: {np.mean(ppls):.6f} ± {np.std(ppls):.4f}  (n={len(ppls)})")
    print("  Expected pretrained PPL: ~2.6250 (diff should be < 0.001)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse SpecDef C@W̃ → W bypass")
    parser.add_argument("--input",  required=True, help="Path to model_specdef.pt")
    parser.add_argument("--output", required=True, help="Output path for fused checkpoint")
    parser.add_argument("--verify", action="store_true",
                        help="Run PPL verification after fusing")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    fuse_specdef(args.input, args.output)

    if args.verify:
        verify_ppl(args.output, args.device)
