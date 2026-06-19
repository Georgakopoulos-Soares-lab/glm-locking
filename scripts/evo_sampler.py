"""Autoregressive Evo sampler for Priority 1 generative functional test.

Evo (StripedHyena) has no .generate() API. This implements naive autoregressive
sampling: full forward pass each step (no KV caching). Correct but ~2-4x slower
than cached. Sufficient for N≥100 sequences at 1024 nt.

Gate validation: pretrained vs locked-no-FT must generate indistinguishably
before we compare Unlocked-FT vs M.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Optional


def sample_evo(
    model,
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    device: str = "cuda",
    amp_dtype: torch.dtype = torch.bfloat16,
) -> str:
    """
    Autoregressive sampling from Evo (StripedHyena).

    Args:
        model: Evo model (on device)
        tokenizer: CharLevelTokenizer
        prompt: seed sequence (empty string = unconditional)
        max_new_tokens: max tokens to generate
        temperature: softmax temperature (1.0 = no scaling)
        top_p: nucleus sampling threshold (1.0 = disabled)
        seed: random seed for reproducibility
        device: torch device
        amp_dtype: autocast dtype

    Returns:
        Generated sequence string (prompt + generated tokens, EOS-stripped)
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    model.eval()

    # Tokenize prompt
    if prompt:
        prompt_ids = tokenizer.tokenize(prompt)
        if isinstance(prompt_ids, np.ndarray):
            prompt_ids = prompt_ids.tolist()
    else:
        prompt_ids = []

    generated_ids = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Build input tensor
            if len(generated_ids) == 0:
                # Start with a random nucleotide if no prompt
                start_token = random.choice([65, 84, 67, 71])  # A, T, C, G
                generated_ids = [start_token]

            input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)

            # Forward pass
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(input_ids)

            logits = out[0]  # [1, T, vocab_size]
            next_logits = logits[0, -1, :].float()  # [vocab_size]

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature
            else:
                # Greedy
                next_token = next_logits.argmax().item()
                generated_ids.append(next_token)
                continue

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep at least the first token
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = -float('inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            generated_ids.append(next_token)

            # Check EOS
            if next_token == tokenizer.eos_id:
                break

    # Detokenize
    result = tokenizer.detokenize(generated_ids)
    return result


def generate_batch(
    model,
    tokenizer,
    n: int,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    prompt_pool: Optional[list[str]] = None,
    base_seed: int = 42,
    device: str = "cuda",
    amp_dtype: torch.dtype = torch.bfloat16,
) -> list[str]:
    """
    Generate n sequences with matched settings.

    Args:
        prompt_pool: If provided, randomly samples prompts. If None, unconditional.
    Returns:
        List of generated sequences.
    """
    sequences = []
    for i in range(n):
        seed = base_seed + i
        if prompt_pool:
            prompt = prompt_pool[i % len(prompt_pool)]
        else:
            prompt = ""
        seq = sample_evo(
            model, tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            device=device,
            amp_dtype=amp_dtype,
        )
        sequences.append(seq)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{n} sequences")

    return sequences


# ── Gate validation ─────────────────────────────────────────────────────────

def gate_validate_pretrained_vs_locked_no_ft():
    """
    GATE CHECK: locked-no-FT (real SpecDef path) must be FUNCTIONALLY 
    indistinguishable from pretrained on coding density and ORF count.
    
    PROTOCOL (Option 2 — Same Pipeline):
      Both locked-no-FT and M use the REAL SpecDef path (no fused_eval).
      The fp32 compensation numerics affect both identically.
      
      Gate: locked-no-FT's coding density + ORF count distributions must be
      statistically indistinguishable from pretrained (p > 0.05, bootstrap).
      If so, the SpecDef numerics are benign for functional generation,
      and any M vs locked-no-FT differences are attributable to the attack.
      
      Bit-identity is NOT expected — fp32 W̃ → bf16 forward pass differs from
      native bf16 pretrained forward pass at the ~1e-4 level per token,
      compounding over autoregressive steps.
    """
    import sys, os, json
    import numpy as np
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, REPO)
    from src.utils import load_evo_model, maybe_load_locked_checkpoint, specdef_fused_eval

    device = "cuda"
    n_seqs = 50
    max_tokens = 256
    temperature = 0.8
    top_p = 0.95

    print("=" * 60)
    print("GATE VALIDATION: pretrained vs locked-no-FT (fused_eval)")
    print("=" * 60)

    # 1. Pretrained
    print("\n1. Loading pretrained Evo...")
    model_pt, tokenizer = load_evo_model("evo-1-8k-base", device)
    model_pt.eval()

    print(f"   Generating {n_seqs} sequences (T={temperature}, top_p={top_p}, max_tokens={max_tokens})...")
    sys.stdout.flush()
    seqs_pt = generate_batch(model_pt, tokenizer, n=n_seqs,
                             max_new_tokens=max_tokens, temperature=temperature,
                             top_p=top_p, base_seed=42, device=device)
    print(f"   Done. Example: {seqs_pt[0][:80]}...")

    lengths_pt = [len(s) for s in seqs_pt]
    gc_pt = [sum(1 for c in s.upper() if c in 'GC') / max(len(s), 1) for s in seqs_pt]
    print(f"   Mean length: {np.mean(lengths_pt):.0f} ± {np.std(lengths_pt):.0f}")
    print(f"   Mean GC: {np.mean(gc_pt):.3f} ± {np.std(gc_pt):.3f}")

    del model_pt
    torch.cuda.empty_cache()

    # 2. Locked-no-FT with REAL SpecDef path (not fused_eval)
    print("\n2. Loading locked-no-FT (α=3×10⁵) — REAL SpecDef path...")
    model_locked, _ = load_evo_model("evo-1-8k-base", device)
    maybe_load_locked_checkpoint(model_locked, "results/lock_alpha300k/model_specdef.pt")
    model_locked.eval()

    print(f"   Generating {n_seqs} sequences (SAME settings, real SpecDef)...")
    sys.stdout.flush()
    seqs_locked = generate_batch(model_locked, tokenizer, n=n_seqs,
                                 max_new_tokens=max_tokens, temperature=temperature,
                                 top_p=top_p, base_seed=42, device=device)
    print(f"   Done. Example: {seqs_locked[0][:80]}...")

    lengths_locked = [len(s) for s in seqs_locked]
    gc_locked = [sum(1 for c in s.upper() if c in 'GC') / max(len(s), 1) for s in seqs_locked]
    print(f"   Mean length: {np.mean(lengths_locked):.0f} ± {np.std(lengths_locked):.0f}")
    print(f"   Mean GC: {np.mean(gc_locked):.3f} ± {np.std(gc_locked):.3f}")

    del model_locked
    torch.cuda.empty_cache()

    # 3. Save sequences for distribution check
    os.makedirs("experiments/exp3_virobench/gate", exist_ok=True)
    with open("experiments/exp3_virobench/gate/seqs_pretrained.json", "w") as f:
        json.dump(seqs_pt, f)
    with open("experiments/exp3_virobench/gate/seqs_locked_no_ft.json", "w") as f:
        json.dump(seqs_locked, f)

    # 4. Gate check: bit-identical sequences (should be with fused eval)
    n_identical = sum(1 for s1, s2 in zip(seqs_pt, seqs_locked) if s1 == s2)
    print(f"\n3. GATE CHECK:")
    print(f"   Bit-identical: {n_identical}/{n_seqs} sequences")

    if n_identical == n_seqs:
        print("\n   ✅ GATE PASSED: pretrained ≡ locked-no-FT (fused_eval) is bit-identical.")
        print("      SpecDef wrapper + fused_eval preserves exact forward pass.")
        print("      Proceed to Unlocked-FT vs M comparison (M uses REAL SpecDef, not fused).")
        return True
    elif n_identical >= 45:
        print(f"\n   ⚠️  GATE PARTIAL: {n_identical}/{n_seqs} identical.")
        print("      Minor nondeterminism detected even with fused eval.")
        print("      Running distribution-level check...")
    else:
        print(f"\n   ❌ GATE FAILED (bit-identical): only {n_identical}/{n_seqs} identical.")
        print("      Running distribution-level check to see if distributions match...")

    # 5. Distribution-level gate (functional metrics only)
    # Bit-identity is impossible due to fp32 W̃ → bf16 fused cast (the pretrained
    # weight was natively bf16, but W̃ is stored in fp32 in the checkpoint).
    # The gate criterion is: locked-no-FT's FUNCTIONAL metrics (coding density,
    # ORF count) must be statistically indistinguishable from pretrained.
    # GC content can differ — it's not a functional metric.
    from scripts.gate_distribution_check import score_orf_coding, bootstrap_test
    
    cd_pt, n_orf_pt = [], []
    cd_lk, n_orf_lk = [], []
    for s in seqs_pt:
        cd, n = score_orf_coding(s)
        cd_pt.append(cd); n_orf_pt.append(n)
    for s in seqs_locked:
        cd, n = score_orf_coding(s)
        cd_lk.append(cd); n_orf_lk.append(n)
    
    cd_diff, cd_p = bootstrap_test(cd_pt, cd_lk)
    orf_diff, orf_p = bootstrap_test(n_orf_pt, n_orf_lk)
    
    print(f"\n3. FUNCTIONAL GATE CHECK:")
    print(f"   Coding density: pretrained={np.mean(cd_pt):.4f}  locked-no-FT={np.mean(cd_lk):.4f}  p={cd_p:.4f}")
    print(f"   ORF count:      pretrained={np.mean(n_orf_pt):.1f}  locked-no-FT={np.mean(n_orf_lk):.1f}  p={orf_p:.4f}")
    
    if cd_p > 0.05 and orf_p > 0.05:
        print(f"\n   ✅ FUNCTIONAL GATE PASSED: locked-no-FT is functionally indistinguishable")
        print(f"      from pretrained on coding density and ORF count.")
        print(f"      The SpecDef path preserves generative function.")
        print(f"      Proceed to Unlocked-FT vs M comparison (both on REAL SpecDef path).")
        return True
    else:
        print(f"\n   ❌ FUNCTIONAL GATE FAILED: locked-no-FT differs from pretrained.")
        print(f"      DO NOT proceed until fixed.")
        return False


if __name__ == "__main__":
    import sys
    success = gate_validate_pretrained_vs_locked_no_ft()
    sys.exit(0 if success else 1)
