"""Direct SVD scaling lock for Evo.

Instead of gradient-based spectral inflation (which tops out at σ_max≈117 after 5000 steps),
this script:
  1. Loads base Evo model
  2. For each target weight matrix, computes top-k singular vectors and directly
     scales those singular values by `scale_factor` (default 100×)
     using the delta formula: W_new = W + (scale-1) * sum_i sigma_i * u_i @ v_i^T
     This preserves ALL other singular values exactly.
  3. Runs `recover_steps` of retain-only finetuning at very low LR to restore
     fluency without significantly deflating the inflated singular values.
  4. Saves checkpoint in the same format as lock.py (partial state_dict of target params).

Compare results to ft_attack_v11_strong_3k (gradient-based approach) to see
if direct spectral inflation produces a stronger convergence barrier.

Usage:
    CUDA_VISIBLE_DEVICES=3 conda run -n evo --no-capture-output \\
        python -u scripts/lock_direct_scale.py --config configs/lock_direct_scale_x100.yaml
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from dataclasses import dataclass, field

import torch
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
    get_lock_targets,
    freeze_all_except,
    compute_svd_stats,
    log_svd_stats,
    load_evo_model,
    save_history_csv,
    count_params,
    count_trainable,
    maybe_enable_gradient_checkpointing,
)


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class DirectScaleConfig:
    run_name: str
    results_dir: str
    model_name: str = "evo-1-8k-base"
    device: str = "cuda:0"
    retain_data_path: str = "data/retain.fasta"
    seed: int = 42
    train_fraction: float = 0.9
    min_seq_len: int = 512

    # Direct scaling
    scale_factor: float = 100.0   # multiply top-k singular values by this
    top_k: int = 5                # how many singular values to inflate

    # Retain recovery (run after scaling to restore fluency)
    recover_steps: int = 500
    recover_lr: float = 1e-6
    batch_size: int = 1
    seq_len: int = 1024
    grad_accum_steps: int = 12
    val_every: int = 50
    val_batches: int = 8
    max_grad_norm: float = 1.0

    target_blocks: set = field(default_factory=lambda: set(range(32)))
    target_layer_patterns: tuple = (
        ".projections.weight",
        ".out_filter_dense.weight",
        ".inner_mha_cls.Wqkv.weight",
        ".inner_mha_cls.out_proj.weight",
        ".mlp.l1.weight",
        ".mlp.l2.weight",
        ".mlp.l3.weight",
    )

    freeze_non_targets: bool = True
    save_checkpoint: bool = True
    use_gradient_checkpointing: bool = True


def _load_config(path: str) -> DirectScaleConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    d.setdefault("results_dir", f"results/{d['run_name']}")
    if "target_blocks" in d:
        val = d["target_blocks"]
        d["target_blocks"] = set(range(val)) if isinstance(val, int) else set(val)
    if "target_layer_patterns" in d:
        d["target_layer_patterns"] = tuple(d["target_layer_patterns"])
    for key in ("recover_lr", "scale_factor"):
        if key in d:
            d[key] = float(d[key])
    from dataclasses import fields as _fields
    valid = {f.name for f in _fields(DirectScaleConfig)}
    d = {k: v for k, v in d.items() if k in valid}
    return DirectScaleConfig(**d)


# ===========================================================================
# Core: direct singular value inflation
# ===========================================================================

def inflate_singular_values(
    target_params: list[torch.Tensor],
    target_names: list[str],
    top_k: int,
    scale_factor: float,
) -> list[dict]:
    """Scale the top-k singular values of each target matrix by scale_factor.

    Uses the delta formula to preserve all non-scaled singular values exactly:
        W_new = W + (scale - 1) * sum_{i=1}^{k} sigma_i * u_i @ v_i^T

    This avoids full SVD reconstruction artifacts and is memory efficient.
    All operations in float32, result cast back to param dtype.
    """
    stats = []
    q = max(2 * top_k + 10, 20)
    with torch.no_grad():
        for name, p in zip(target_names, target_params):
            w = p.data.float()
            U, S, V = torch.svd_lowrank(w, q=q, niter=4)
            sigma_before = S[:top_k].clone()

            # Delta: add (scale-1) * sigma_i * u_i @ v_i^T for each top-k component
            delta = torch.zeros_like(w)
            for i in range(top_k):
                delta.add_(U[:, i:i+1] @ V[:, i:i+1].T, alpha=float((scale_factor - 1.0) * S[i].item()))

            p.data.add_(delta.to(p.dtype))

            # Verify new sigma (recompute with updated weights)
            _, S_new, _ = torch.svd_lowrank(p.data.float(), q=q, niter=4)
            sigma_after = S_new[:top_k].clone()

            stats.append({
                "name": name,
                "sigma_1_before": float(sigma_before[0]),
                "sigma_1_after": float(sigma_after[0]),
                "ratio": float(sigma_after[0] / sigma_before[0]) if sigma_before[0] > 0 else float("inf"),
            })
            print(f"  {name[-50:]:50s}  σ1: {sigma_before[0]:.3f} → {sigma_after[0]:.3f}  ({sigma_after[0]/max(sigma_before[0],1e-8):.1f}×)")

    return stats


# ===========================================================================
# Retain recovery
# ===========================================================================

def retain_recovery(
    model,
    tokenizer,
    train_seqs,
    val_seqs,
    target_names_set: set,
    cfg: DirectScaleConfig,
    amp_dtype,
) -> list[dict]:
    """Fine-tune with pure retain loss (no spectral term) to restore fluency."""
    from torch.optim import SGD

    print(f"\nStarting retain recovery: {cfg.recover_steps} steps, lr={cfg.recover_lr}")
    print("  Optimizer: SGD with momentum=0.9 (predictable, no adaptive deflation)")

    # Freeze non-targets
    freeze_all_except(model, target_names_set)
    trainable_list = [p for p in model.parameters() if p.requires_grad]

    optimizer = SGD(trainable_list, lr=cfg.recover_lr, momentum=0.9)
    history = []
    optimizer.zero_grad(set_to_none=True)

    for step in range(cfg.recover_steps):
        model.train()
        step_loss = 0.0

        for _ in range(cfg.grad_accum_steps):
            batch = build_batch(tokenizer, train_seqs, cfg.batch_size, cfg.seq_len, cfg.device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, _ = model(batch)
                loss = causal_lm_loss(logits, batch) / cfg.grad_accum_steps
            step_loss += loss.detach().float().item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable_list, cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.val_every == 0 or step == cfg.recover_steps - 1:
            val_loss, val_ppl, val_acc = evaluate(
                model, tokenizer, val_seqs, cfg.device,
                cfg.val_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
            )
            record = {
                "step": step,
                "train_loss": round(step_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 4),
                "val_acc": round(val_acc, 4),
            }
            history.append(record)
            print(f"  Recover step {step:04d} | train={step_loss:.4f} | val={val_loss:.4f} | ppl={val_ppl:.4f}")

    return history


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    os.makedirs(cfg.results_dir, exist_ok=True)
    set_seed(cfg.seed)
    amp_dtype, _ = get_amp_settings()

    print(f"=== Direct SVD Scaling Lock: {cfg.run_name} ===")
    print(f"  scale_factor={cfg.scale_factor}x, top_k={cfg.top_k}")
    print(f"  recover_steps={cfg.recover_steps}, recover_lr={cfg.recover_lr}")

    # --- Data ---
    print("\nLoading retain data...")
    sequences = load_sequences(cfg.retain_data_path, cfg.min_seq_len)
    train_seqs, val_seqs = split_sequences(sequences, cfg.train_fraction)
    print(f"  {len(sequences)} seqs → {len(train_seqs)} train / {len(val_seqs)} val")

    # --- Model ---
    model, tokenizer = load_evo_model(cfg.model_name, cfg.device)
    maybe_enable_gradient_checkpointing(model, cfg.use_gradient_checkpointing)

    total = count_params(model)
    print(f"\nTotal params: {total:,}")

    # --- Targets ---
    target_names, target_params = get_lock_targets(
        model, cfg.target_blocks, cfg.target_layer_patterns
    )
    if not target_params:
        raise RuntimeError("No target parameters matched. Check target_layer_patterns.")
    print(f"Target matrices: {len(target_params)}")

    target_names_set = set(target_names)

    # --- SVD stats before ---
    svd_csv = os.path.join(cfg.results_dir, "svd_stats.csv")
    pre_stats = compute_svd_stats(target_names, target_params, top_k=max(cfg.top_k, 3))
    log_svd_stats(pre_stats, "before_scaling", svd_csv)

    sigma_mean_before = sum(s["sigma_1"] for s in pre_stats) / len(pre_stats)
    sigma_max_before = max(s["sigma_1"] for s in pre_stats)
    print(f"\nSigma_1 before: mean={sigma_mean_before:.3f}, max={sigma_max_before:.3f}")

    # --- Baseline val_loss before scaling ---
    print("\nBaseline val_loss BEFORE scaling:")
    baseline_loss, baseline_ppl, baseline_acc = evaluate(
        model, tokenizer, val_seqs, cfg.device,
        cfg.val_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
    )
    print(f"  val_loss={baseline_loss:.4f}  ppl={baseline_ppl:.4f}  acc={baseline_acc:.4f}")

    # --- Direct singular value inflation ---
    print(f"\n=== Inflating top-{cfg.top_k} singular values by {cfg.scale_factor}x ===")
    inflation_stats = inflate_singular_values(target_params, target_names, cfg.top_k, cfg.scale_factor)

    # SVD stats immediately after scaling
    post_scale_stats = compute_svd_stats(target_names, target_params, top_k=max(cfg.top_k, 3))
    log_svd_stats(post_scale_stats, "after_scaling", svd_csv)
    sigma_mean_after = sum(s["sigma_1"] for s in post_scale_stats) / len(post_scale_stats)
    sigma_max_after = max(s["sigma_1"] for s in post_scale_stats)
    print(f"\nSigma_1 after scaling: mean={sigma_mean_after:.3f}, max={sigma_max_after:.3f}")

    # Val_loss immediately after scaling (should be high — model is partially broken)
    print("\nVal_loss AFTER scaling (before recovery):")
    post_scale_loss, post_scale_ppl, _ = evaluate(
        model, tokenizer, val_seqs, cfg.device,
        cfg.val_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
    )
    print(f"  val_loss={post_scale_loss:.4f}  ppl={post_scale_ppl:.4f}")

    # --- Retain recovery ---
    recovery_history = []
    if cfg.recover_steps > 0:
        recovery_history = retain_recovery(
            model, tokenizer, train_seqs, val_seqs,
            target_names_set, cfg, amp_dtype,
        )
        save_history_csv(recovery_history, os.path.join(cfg.results_dir, "recovery_metrics.csv"))

    # --- SVD stats after recovery ---
    post_recover_stats = compute_svd_stats(target_names, target_params, top_k=max(cfg.top_k, 3))
    log_svd_stats(post_recover_stats, "after_recovery", svd_csv)
    sigma_mean_final = sum(s["sigma_1"] for s in post_recover_stats) / len(post_recover_stats)
    sigma_max_final = max(s["sigma_1"] for s in post_recover_stats)
    print(f"\nSigma_1 after recovery: mean={sigma_mean_final:.3f}, max={sigma_max_final:.3f}")
    print(f"  (before scaling: mean={sigma_mean_before:.3f}, max={sigma_max_before:.3f})")
    print(f"  Inflation retained: {sigma_mean_final/max(sigma_mean_before,1e-8):.1f}x mean, {sigma_max_final/max(sigma_max_before,1e-8):.1f}x max")

    print("\nFinal val_loss AFTER recovery:")
    final_loss, final_ppl, final_acc = evaluate(
        model, tokenizer, val_seqs, cfg.device,
        cfg.val_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
    )
    print(f"  val_loss={final_loss:.4f}  ppl={final_ppl:.4f}  acc={final_acc:.4f}")
    print(f"  Baseline was: {baseline_loss:.4f} — delta: {final_loss - baseline_loss:+.4f}")

    # --- Save summary ---
    summary_path = os.path.join(cfg.results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Run: {cfg.run_name}\n")
        f.write(f"Scale factor: {cfg.scale_factor}x\n")
        f.write(f"Top-k: {cfg.top_k}\n")
        f.write(f"Recover steps: {cfg.recover_steps}\n")
        f.write(f"Recover LR: {cfg.recover_lr}\n")
        f.write(f"Target matrices: {len(target_params)}\n\n")
        f.write(f"Sigma_1 before: mean={sigma_mean_before:.3f}, max={sigma_max_before:.3f}\n")
        f.write(f"Sigma_1 after scale: mean={sigma_mean_after:.3f}, max={sigma_max_after:.3f}\n")
        f.write(f"Sigma_1 after recovery: mean={sigma_mean_final:.3f}, max={sigma_max_final:.3f}\n")
        f.write(f"Net inflation retained: {sigma_mean_final/max(sigma_mean_before,1e-8):.1f}x mean\n\n")
        f.write(f"Val loss baseline: {baseline_loss:.4f}\n")
        f.write(f"Val loss post-scale: {post_scale_loss:.4f}\n")
        f.write(f"Val loss final: {final_loss:.4f}\n")

    # --- Save checkpoint ---
    if cfg.save_checkpoint:
        ckpt_path = os.path.join(cfg.results_dir, "model_locked.pt")
        if cfg.freeze_non_targets:
            # Re-freeze to identify target params, then save only those
            freeze_all_except(model, target_names_set)
            trained_sd = {n: p.data for n, p in model.named_parameters() if n in target_names_set}
        else:
            trained_sd = model.state_dict()
        torch.save(trained_sd, ckpt_path)
        print(f"\nSaved checkpoint ({len(trained_sd)} tensors): {ckpt_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
