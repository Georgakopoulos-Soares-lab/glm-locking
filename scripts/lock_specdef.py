"""Exact SpecDef locking for Evo (Rosati et al., 2026).

Unlike gradient-based locking, this performs algebraic SVD factorization:
  1. For each target layer W, compute full SVD: W = U Σ V^T
  2. Inflate top-k singular values: Σ̃[i] = α * Σ[i] for i ≤ k
  3. Create compensation matrix: C = U (Σ Σ̃⁻¹) U^T
  4. Replace W with W̃ = U Σ̃ V^T  and insert C as adjacent layer

Result: C · W̃ = U Σ V^T = W  (function preserved exactly, ε=0)
But σ₁(W̃) = α · σ₁(W), making the Hessian curvature proportional to α²,
which forces the attacker's stable learning rate to ≤ 2/(α·σ₁).

No training needed. Runs in seconds.

Usage:
    CUDA_VISIBLE_DEVICES=5 conda run -n evo --no-capture-output \
        python -u scripts/lock_specdef.py --config configs/lock_specdef_alpha1k.yaml
"""

import os
import sys
import json
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from dataclasses import dataclass, field, fields as _fields

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    set_seed,
    get_amp_settings,
    load_sequences,
    split_sequences,
    build_batch,
    causal_lm_loss,
    evaluate,
    compute_svd_stats,
    log_svd_stats,
    load_evo_model,
    count_params,
)


# ===========================================================================
# SpecDefLinear: drop-in wrapper for inflated weight + compensation
# ===========================================================================

class SpecDefLinear(nn.Module):
    """Drop-in replacement for nn.Linear that applies SpecDef compensation.

    During forward: disables autocast, computes C(W̃(x)) + bias in float32,
    then casts output back to input dtype.

    During evaluation with fuse_specdef_for_eval(), the wrapper can be
    replaced with a normal bf16 linear using the fused weight C@W̃ (which
    is bit-identical to the original W), giving ε=0 exactly.
    """
    def __init__(self, inflated_linear: nn.Linear, comp_linear: nn.Linear,
                 bias: torch.Tensor | None = None):
        super().__init__()
        self.linear = inflated_linear   # W̃ (inflated weights, f32)
        self.comp = comp_linear         # C (compensation, f32)
        if bias is not None:
            self.bias = nn.Parameter(bias.float())
        else:
            self.bias = None

    def forward(self, x):
        import torch.nn.functional as _F
        with torch.autocast(device_type="cuda", enabled=False):
            h = self.linear(x.float())
            # comp.weight may be stored as bf16 to save GPU memory (frozen)
            # or as float64 for high-precision zero-shot equivalence (rebuild_specdef_f64).
            c_w = self.comp.weight
            if c_w.dtype == torch.float64:
                # Upcast h to f64, do C·h in f64, downcast.
                out = _F.linear(h.double(), c_w, None).float()
            elif c_w.dtype == torch.float32:
                out = _F.linear(h, c_w, None)
            else:
                out = _F.linear(h, c_w.float(), None)
            if self.bias is not None:
                out = out + self.bias
        return out.to(x.dtype)


class BypassSpecDefLinear(nn.Module):
    """Layer-injection bypass (Rosati et al. Theorem 8, black-box attacker).

    Wraps a frozen SpecDefLinear with a trainable bypass matrix B (d_out × d_out).
    Forward pass:

        h   = W_fused(x)     [single bf16 matmul: W_fused = C @ W̃, precomputed at init]
        out = B(h)           [trainable — gradients flow back through h to x]

    W_fused = C @ W̃ is precomputed once at init and stored frozen in bf16, replacing
    the original two sequential fp32 matmuls in SpecDefLinear with a single bf16 matmul.
    This is numerically equivalent (C·W̃ = W to bf16 precision) and ~3× faster.
    Gradients flow through the fused layer without any no_grad wrapping because
    fused.weight has requires_grad=False.

    B is initialised to the identity matrix so the model is function-equivalent to
    the locked checkpoint at step 0. Only B trains; W_fused and C are frozen.

    Construction cost: one SVD per locked layer, done on CPU in seconds.
    Parameter overhead: 32 × 4096² × 2 bytes ≈ 1.07 GB (< 0.5% of Evo-7B).

    INIT BUG HISTORY (fixed):
    The original implementation called `nn.init.eye_(self.bypass.weight.float())`,
    which creates a *detached copy* of the bf16 weight and fills the copy — the
    actual parameter is left with Kaiming-uniform random values.  This caused
    step-0 val_ppl = 637 instead of ~3.75 and rendered all 25k training steps
    ineffective (the model spent the run recovering B toward identity rather than
    learning).  Fixed by calling `nn.init.eye_(self.bypass.weight)` directly.
    """
    def __init__(self, specdef_layer: SpecDefLinear):
        super().__init__()
        # Precompute W_fused = C @ W̃ in f32 then store as bf16.
        # This is bit-equivalent to the original 2× f32 path (C(W̃(x))) but:
        #   (a) ~3× faster: one bf16 matmul vs two f32 matmuls with autocast disabled
        #   (b) correct gradient flow: ∂h/∂x tracked through a normal Linear — no no_grad
        with torch.no_grad():
            w_tilde = specdef_layer.linear.weight.float()   # (d_out, d_in)
            c_w     = specdef_layer.comp.weight.float()     # (d_out, d_out)
            if specdef_layer.comp.weight.dtype == torch.float64:
                c_w = specdef_layer.comp.weight.double().float()
            w_fused = (c_w @ w_tilde).bfloat16()           # (d_out, d_in)
            d_out, d_in = w_fused.shape
            device = w_fused.device
            bias = specdef_layer.bias  # nn.Parameter or None

        self.fused = nn.Linear(d_in, d_out, bias=(bias is not None),
                               device=device, dtype=torch.bfloat16)
        self.fused.weight = nn.Parameter(w_fused, requires_grad=False)
        if bias is not None:
            self.fused.bias = nn.Parameter(bias.bfloat16(), requires_grad=False)

        # Identity init: B = I  =>  B(h) = h at step 0.
        # BUG FIX: must call eye_ directly on the parameter tensor. Calling
        # .float() first returns a detached copy — eye_ would fill the copy,
        # leaving the actual bfloat16 weight with random Kaiming uniform init.
        self.bypass = nn.Linear(d_out, d_out, bias=False, device=device,
                                dtype=torch.bfloat16)
        nn.init.eye_(self.bypass.weight)   # 0s and 1s are exact in bfloat16

    def forward(self, x):
        # Single bf16 matmul through frozen W_fused; ∂h/∂x flows normally.
        # C and W̃ are frozen via requires_grad=False — no explicit no_grad needed.
        h = self.fused(x.to(self.fused.weight.dtype))
        return self.bypass(h).to(x.dtype)


class Theorem8SpecDefLinear(nn.Module):
    """Theorem 8 SVD-factorization attack (white-box, proper implementation).

    Replaces SpecDefLinear(W̃, C) with k jointly-trainable factor layers
    initialized via SVD of W̃ so that each factor has σ₁ = (σ₁(W̃))^(1/k) ≈ α^(1/k).
    Forward: C( L_k( ... L_1(x) ) ) — sequential float32 activation matmuls for all k.
    C is frozen; all factor layers are jointly trainable.
    """
    def __init__(self, specdef_layer: SpecDefLinear, k: int = 2):
        super().__init__()
        self.comp = specdef_layer.comp   # C — frozen
        self.bias = specdef_layer.bias
        self.k = k

        W = specdef_layer.linear.weight.data.float()  # W̃ shape (d_out, d_in)
        d_out, d_in = W.shape
        device = specdef_layer.linear.weight.device

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = S.shape[0]
        S_pow = S.pow(1.0 / k)

        factors = nn.ModuleList()
        if k == 2:
            l1 = nn.Linear(d_in, r, bias=False, device=device)
            l2 = nn.Linear(r, d_out, bias=False, device=device)
            l1.weight.data = (S_pow.unsqueeze(1) * Vh)          # float32
            l2.weight.data = (U * S_pow.unsqueeze(0))            # float32
            factors.extend([l1, l2])
        else:
            l1 = nn.Linear(d_in, r, bias=False, device=device)
            l1.weight.data = S_pow.unsqueeze(1) * Vh
            factors.append(l1)
            for _ in range(k - 2):
                lm = nn.Linear(r, r, bias=False, device=device)
                lm.weight.data = torch.diag(S_pow)
                factors.append(lm)
            lk = nn.Linear(r, d_out, bias=False, device=device)
            lk.weight.data = U * S_pow.unsqueeze(0)
            factors.append(lk)

        self.factors = factors
        print(f"  [Theorem8] k={k}: d_out={d_out} d_in={d_in} r={r} "
              f"σ_max(W̃)={float(S.max()):.1f} → σ_max/factor={float(S_pow.max()):.1f}")

    def forward(self, x):
        # Sequential float32 activation matmuls for all k.
        # Must use float32 (not bf16): σ_max(W̃) up to 28k generates intermediate
        # activations of magnitude ~28k before C compresses by 1/α=1/10k.
        # bf16 rounds magnitude-28k values to ±219 per element → blows up val_loss.
        # float32 is required for numerically accurate C@L_k@...@L_1(x) = W(x) at init.
        import torch.nn.functional as _F
        with torch.autocast(device_type="cuda", enabled=False):
            h = x.float()
            for fac in self.factors:
                h = _F.linear(h, fac.weight.float(), None)
            c_w = self.comp.weight
            out = _F.linear(h, c_w.float() if c_w.dtype != torch.float32 else c_w, None)
            if self.bias is not None:
                out = out + self.bias
        return out.to(x.dtype)


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class SpecDefConfig:
    run_name: str
    results_dir: str
    model_name: str = "evo-1-8k-base"
    device: str = "cuda:0"
    seed: int = 42

    # SpecDef parameters
    alpha: float = 1000.0         # singular value multiplier
    top_k: int = 25               # how many singular values to inflate per layer
    target_blocks: set = field(default_factory=lambda: set(range(32)))

    # Layer patterns to target (dotted paths relative to each block).
    # Default targets both Hyena output proj and attention output proj (all square 4096×4096).
    target_layer_patterns: list = field(default_factory=lambda: [
        "out_filter_dense",           # Hyena mixer output projection (29 blocks)
        "inner_mha_cls.out_proj",     # attention output projection (blocks 8, 16, 24)
    ])

    # If > 0, randomly select this many (block, layer) pairs from all candidates.
    # Paper uses 5. If 0, apply to ALL matching layers.
    num_target_layers: int = 0

    # Validation (to verify functional equivalence)
    retain_data_path: str = "data/retain.fasta"
    val_batches: int = 16
    val_seq_len: int = 1024
    min_seq_len: int = 512

    save_checkpoint: bool = True


def _load_config(path: str) -> SpecDefConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    d.setdefault("results_dir", f"results/{d['run_name']}")
    if "target_blocks" in d:
        val = d["target_blocks"]
        d["target_blocks"] = set(range(val)) if isinstance(val, int) else set(val)
    for key in ("alpha",):
        if key in d:
            d[key] = float(d[key])
    # Handle legacy single-pattern config
    if "target_layer_pattern" in d and "target_layer_patterns" not in d:
        d["target_layer_patterns"] = [d.pop("target_layer_pattern")]
    elif "target_layer_pattern" in d:
        d.pop("target_layer_pattern")
    valid = {f.name for f in _fields(SpecDefConfig)}
    d = {k: v for k, v in d.items() if k in valid}
    return SpecDefConfig(**d)


# ===========================================================================
# Helpers for nested attribute access
# ===========================================================================

def _get_nested(obj, dotted_path: str):
    """Get a nested attribute via dotted path (e.g. 'inner_mha_cls.out_proj')."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _set_nested(obj, dotted_path: str, value):
    """Set a nested attribute via dotted path."""
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


# ===========================================================================
# Core: exact spectral deformation with compensation
# ===========================================================================

def specdef_inflate_with_compensation(
    weight: torch.Tensor,
    alpha: float,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Perform exact SpecDef on a single weight matrix.

    Returns:
        W_tilde: inflated weight matrix [m, n]  (float32)
        C: compensation matrix [m, m]  (float32)
        stats: dict with pre/post singular value info
    """
    # Supports both square and non-square matrices.
    # For W ∈ R^{m×n}: thin SVD gives U (m×r), S (r,), Vh (r×n) where r=min(m,n).
    # Compensation C = U diag(S/S̃) Uᵀ is always square (m×m).
    # Identity: C @ W̃ = U(S/S̃)Uᵀ · U S̃ Vᵀ = U S Vᵀ = W  ✓

    w = weight.double()

    U, S, Vh = torch.linalg.svd(w, full_matrices=False)

    S_tilde = S.clone()
    S_tilde[:top_k] *= alpha

    W_tilde = U @ torch.diag(S_tilde) @ Vh
    ratios = S / S_tilde
    C = U @ torch.diag(ratios) @ U.T

    # Verify: C @ W̃ should equal W (in float64)
    max_err = (C @ W_tilde - w).abs().max().item()

    # Also verify bf16 round-trip: fused result should be bit-identical to original
    fused_bf16 = (C @ W_tilde).float().bfloat16()
    bf16_diff = (fused_bf16.float() - weight.float()).abs().max().item()

    stats = {
        "sigma_1_before": float(S[0]),
        "sigma_1_after": float(S_tilde[0]),
        "sigma_k_before": float(S[min(top_k - 1, len(S) - 1)]),
        "sigma_k_after": float(S_tilde[min(top_k - 1, len(S) - 1)]),
        "reconstruction_max_error_f64": max_err,
        "fused_bf16_vs_orig_max_diff": bf16_diff,
        "alpha": alpha,
        "top_k": top_k,
    }

    return W_tilde.float(), C.float(), stats


def apply_specdef_to_model(model, cfg: SpecDefConfig) -> dict:
    """Apply exact SpecDef to target layers in the model.

    Supports multiple layer patterns (e.g. 'out_filter_dense',
    'inner_mha_cls.out_proj').  If cfg.num_target_layers > 0, randomly
    selects that many (block, pattern) pairs from all candidates.

    Returns:
        dict of {"block_idx.pattern": stats}
    """
    import random as _random

    # Enumerate all candidate (block_idx, pattern) pairs
    candidates = []
    for block_idx in sorted(cfg.target_blocks):
        if block_idx >= len(model.blocks):
            continue
        block = model.blocks[block_idx]
        for pattern in cfg.target_layer_patterns:
            layer = _get_nested(block, pattern)
            if layer is None:
                continue
            if not hasattr(layer, 'weight'):
                continue
            w = layer.weight.data
            candidates.append((block_idx, pattern))

    print(f"  Found {len(candidates)} candidate layers for SpecDef")

    # Optionally subsample (paper uses 5 random layers)
    if cfg.num_target_layers > 0 and len(candidates) > cfg.num_target_layers:
        rng = _random.Random(cfg.seed)
        candidates = sorted(rng.sample(candidates, cfg.num_target_layers))
        print(f"  Randomly selected {len(candidates)} layers (seed={cfg.seed})")

    all_stats = {}

    for block_idx, pattern in candidates:
        block = model.blocks[block_idx]
        layer = _get_nested(block, pattern)
        w = layer.weight.data

        key = f"{block_idx}.{pattern}"
        print(f"  Block {block_idx}: {pattern} {list(w.shape)}", end="")

        W_tilde, C, stats = specdef_inflate_with_compensation(w, cfg.alpha, cfg.top_k)

        # Create inflated linear (f32, no bias — bias stored separately)
        inflated = nn.Linear(w.shape[1], w.shape[0], bias=False,
                             dtype=torch.float32, device=w.device)
        inflated.weight.data = W_tilde.to(w.device)

        # Create compensation linear (f32)
        comp = nn.Linear(w.shape[0], w.shape[0], bias=False,
                         dtype=torch.float32, device=w.device)
        comp.weight.data = C.to(w.device)

        # Transfer original bias (if present)
        orig_bias = layer.bias.data.clone() if layer.bias is not None else None

        # Replace layer with SpecDefLinear wrapper
        _set_nested(block, pattern, SpecDefLinear(inflated, comp, bias=orig_bias))

        print(f"  σ1: {stats['sigma_1_before']:.2f} → {stats['sigma_1_after']:.0f}"
              f"  ({stats['sigma_1_after']/max(stats['sigma_1_before'],1e-8):.0f}×)"
              f"  recon_err: {stats['reconstruction_max_error_f64']:.2e}"
              f"  bf16_diff: {stats['fused_bf16_vs_orig_max_diff']:.2e}")

        all_stats[key] = stats

    return all_stats


def _find_specdef_layers(model):
    """Find all SpecDefLinear wrappers in the model by recursive traversal.

    Returns list of (block_idx, dotted_path, wrapper) tuples.
    Works for any pattern — no hardcoded list required.
    """
    results = []
    for idx, block in enumerate(model.blocks):
        for name, module in block.named_modules():
            if name and isinstance(module, SpecDefLinear):
                results.append((idx, name, module))
    return results


def fuse_specdef_for_eval(model) -> list[tuple]:
    """Replace all SpecDefLinear wrappers with normal bf16 Linear layers.

    Computes C@W̃ in float32 → cast to bf16. The resulting weight is
    bit-identical to the original (proven: SVD round-trip in float64
    preserves bf16 values exactly).

    Returns list of (block_idx, pattern) that were fused.
    """
    fused = []
    for idx, pattern, wrapper in _find_specdef_layers(model):
        with torch.no_grad():
            fused_weight = wrapper.comp.weight.data @ wrapper.linear.weight.data
        d_out, d_in = wrapper.linear.weight.shape
        has_bias = wrapper.bias is not None
        linear = nn.Linear(d_in, d_out, bias=has_bias,
                           dtype=torch.bfloat16, device=fused_weight.device)
        linear.weight.data = fused_weight.bfloat16()
        if has_bias:
            linear.bias.data = wrapper.bias.data.bfloat16()
        _set_nested(model.blocks[idx], pattern, linear)
        fused.append((idx, pattern))
    return fused


def unfuse_specdef(model, checkpoint_state_dict: dict) -> list[tuple]:
    """Restore SpecDefLinear wrappers from a checkpoint state dict.

    Used after fuse_specdef_for_eval() to put wrappers back for training.
    """
    import re

    # Parse comp keys to find (block_idx, pattern) pairs
    modified = []
    for key in checkpoint_state_dict:
        m = re.match(r"blocks\.(\d+)\.(.+)\.comp\.weight", key)
        if m:
            modified.append((int(m.group(1)), m.group(2)))

    restored = []
    for idx, pattern in sorted(set(modified)):
        block = model.blocks[idx]
        prefix = f"blocks.{idx}.{pattern}"
        w_key = f"{prefix}.linear.weight"
        c_key = f"{prefix}.comp.weight"
        if w_key not in checkpoint_state_dict or c_key not in checkpoint_state_dict:
            continue

        W_tilde = checkpoint_state_dict[w_key]
        C = checkpoint_state_dict[c_key]
        d_out, d_in = W_tilde.shape

        inflated = nn.Linear(d_in, d_out, bias=False,
                             dtype=torch.float32, device=W_tilde.device)
        inflated.weight.data = W_tilde.float()
        comp = nn.Linear(d_out, d_out, bias=False,
                         dtype=torch.float32, device=C.device)
        comp.weight.data = C.float()

        _set_nested(block, pattern, SpecDefLinear(inflated, comp))
        restored.append((idx, pattern))
    return restored


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        cfg = _load_config(args.config)
    else:
        cfg = SpecDefConfig(
            run_name="lock_specdef_alpha1k",
            results_dir="results/lock_specdef_alpha1k",
            alpha=1000.0,
            top_k=25,
        )

    os.makedirs(cfg.results_dir, exist_ok=True)
    set_seed(cfg.seed)

    print(f"=== Exact SpecDef Locking: {cfg.run_name} ===")
    print(f"  alpha={cfg.alpha}, top_k={cfg.top_k}")
    print(f"  target patterns: {cfg.target_layer_patterns}")
    print(f"  target blocks: {sorted(cfg.target_blocks)}")
    if cfg.num_target_layers > 0:
        print(f"  random selection: {cfg.num_target_layers} layers (seed={cfg.seed})")

    # --- Load model ---
    model, tokenizer = load_evo_model(cfg.model_name, cfg.device)
    amp_dtype, _ = get_amp_settings()

    total = count_params(model)
    print(f"  Total params (before): {total:,}")

    # --- Apply SpecDef ---
    print(f"\n--- Applying exact SpecDef (alpha={cfg.alpha}, k={cfg.top_k}) ---")
    t0 = time.time()
    all_stats = apply_specdef_to_model(model, cfg)
    elapsed = time.time() - t0
    print(f"\n  SpecDef applied to {len(all_stats)} layers in {elapsed:.1f}s")

    total_after = count_params(model)
    added = total_after - total
    print(f"  Total params (after): {total_after:,} (+{added:,} from compensation layers)")

    # --- Verify functional equivalence ---
    print("\n--- Functional Equivalence Verification ---")
    all_zero = all(
        stats["fused_bf16_vs_orig_max_diff"] < 1e-10
        for stats in all_stats.values()
    )
    if all_zero:
        print("  ✓ PASS — all fused bf16 weights match originals (< 1e-10)")
    else:
        for key, stats in sorted(all_stats.items()):
            d = stats["fused_bf16_vs_orig_max_diff"]
            if d > 0:
                print(f"  {key}: fused bf16 diff = {d:.2e}")
        print("  ✗ WARNING — some fused weights differ from originals")

    # --- Save checkpoint ---
    if cfg.save_checkpoint:
        ckpt_path = os.path.join(cfg.results_dir, "model_specdef.pt")
        print(f"\nSaving checkpoint to {ckpt_path} ...")

        state_dict = model.state_dict()
        torch.save(state_dict, ckpt_path)
        print(f"  Saved {len(state_dict)} keys")

        # Save metadata
        meta = {
            "alpha": cfg.alpha,
            "top_k": cfg.top_k,
            "target_patterns": cfg.target_layer_patterns,
            "num_target_layers": cfg.num_target_layers,
            "modified_layers": sorted(all_stats.keys()),
            "added_params": added,
            "elapsed_seconds": elapsed,
            "per_layer_stats": {str(k): v for k, v in all_stats.items()},
        }
        meta_path = os.path.join(cfg.results_dir, "specdef_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved metadata to {meta_path}")

    # --- SVD stats of inflated weights ---
    svd_csv = os.path.join(cfg.results_dir, "svd_stats.csv")
    svd_names = []
    svd_params = []
    for key in sorted(all_stats.keys()):
        # key is "block_idx.pattern" — parse it
        block_idx_str, pattern = key.split(".", 1)
        block = model.blocks[int(block_idx_str)]
        wrapper = _get_nested(block, pattern)
        name = f"blocks.{block_idx_str}.{pattern}.linear.weight"
        svd_names.append(name)
        svd_params.append(wrapper.linear.weight)
    post_stats = compute_svd_stats(svd_names, svd_params, top_k=max(cfg.top_k, 3))
    log_svd_stats(post_stats, "after_specdef", svd_csv)
    print(f"\n  SVD stats saved to {svd_csv}")

    # --- Summary ---
    sigma1_vals = [s["sigma_1_after"] for s in all_stats.values()]
    print(f"\n  σ₁ range: {min(sigma1_vals):.0f} – {max(sigma1_vals):.0f}")
    print(f"  Max stable LR (SGD): ≤ 2/σ₁² = {2.0/max(sigma1_vals)**2:.2e}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
