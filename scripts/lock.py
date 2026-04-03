"""SpecDef locking for Evo — inflate top-k singular values of all linear layers
in target Hyena blocks while preserving model utility via a retain loss.

Usage:
    python scripts/lock.py
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
from contextlib import nullcontext

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    LockConfig,
    set_seed,
    get_amp_settings,
    setup_ddp,
    cleanup_ddp,
    wrap_ddp,
    load_sequences,
    split_sequences,
    build_batch,
    causal_lm_loss,
    next_token_accuracy,
    evaluate,
    get_lock_targets,
    freeze_all_except,
    compute_svd_stats,
    log_svd_stats,
    spectral_term_topk,
    get_alpha,
    load_evo_model,
    save_history_csv,
    count_params,
    count_trainable,
    count_params_by_name,
)

# ===========================================================================
# Config — edit here or override via --config <yaml>
# ===========================================================================
CONFIG = LockConfig(
    run_name="lock_v10",
    results_dir="results/lock_v10",

    retain_data_path="data/retain.fasta",
    model_name="evo-1-8k-base",
    device="cuda:0",
    seed=42,
    train_fraction=0.9,
    min_seq_len=512,

    lock_steps=1000,
    lock_lr=5e-5,
    alpha_start=0.8,
    alpha_end=0.3,
    top_k=3,
    batch_size=8,
    seq_len=1024,
    grad_accum_steps=1,
    val_every=50,
    val_batches=8,
    max_grad_norm=1.0,

    # Lock ALL 32 blocks — attacker will also attack all 32
    target_blocks=set(range(32)),
    target_layer_patterns=(
        ".projections.weight",
        ".out_filter_dense.weight",
        ".mlp.l1.weight",
        ".mlp.l2.weight",
        ".mlp.l3.weight",
    ),

    save_checkpoint=True,
)


# ===========================================================================
# YAML config loader
# ===========================================================================
def _load_config(path: str) -> LockConfig:
    """Load a LockConfig from a YAML file."""
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f)
    # Auto-derive results_dir from run_name — keeps configs DRY
    d.setdefault("results_dir", f"results/{d['run_name']}")
    if "target_blocks" in d:
        val = d["target_blocks"]
        d["target_blocks"] = set(range(val)) if isinstance(val, int) else set(val)
    if "target_layer_patterns" in d:
        d["target_layer_patterns"] = tuple(d["target_layer_patterns"])
    # YAML loads scientific notation as string (e.g. '5e-5') — cast to float
    for key in ("lock_lr", "lr"):
        if key in d:
            d[key] = float(d[key])
    # Drop keys not in the dataclass (e.g. use_gradient_checkpointing in old configs)
    from dataclasses import fields as _fields
    valid = {f.name for f in _fields(LockConfig)}
    d = {k: v for k, v in d.items() if k in valid}
    return LockConfig(**d)


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    args, _ = parser.parse_known_args()

    # --- DDP setup ---
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)

    cfg = _load_config(args.config) if args.config else CONFIG
    cfg.device = f"cuda:{local_rank}"
    if is_main:
        os.makedirs(cfg.results_dir, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()

    # Deterministic split (same across ranks), then offset seed for data diversity
    set_seed(cfg.seed)
    amp_dtype, use_scaler = get_amp_settings()

    # --- Data ---
    if is_main:
        print("Loading retain data...")
    sequences = load_sequences(cfg.retain_data_path, cfg.min_seq_len)
    train_seqs, val_seqs = split_sequences(sequences, cfg.train_fraction)
    if is_main:
        print(f"  {len(sequences)} sequences -> {len(train_seqs)} train / {len(val_seqs)} val")

    # Re-seed so each rank samples different batches
    set_seed(cfg.seed + rank)

    # --- Model ---
    model, tokenizer = load_evo_model(cfg.model_name, cfg.device)

    # --- Targets ---
    target_names, target_params = get_lock_targets(
        model, cfg.target_blocks, cfg.target_layer_patterns
    )
    if not target_params:
        raise RuntimeError("No target parameters matched. Check target_layer_patterns.")

    target_names_set = set(target_names)
    freeze_all_except(model, target_names_set)

    # --- DDP wrap (target_params still references raw model weights) ---
    model, raw_model = wrap_ddp(model, local_rank)

    total = count_params(raw_model)
    trainable = count_trainable(raw_model)
    if is_main:
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
        print(f"Target matrices:  {len(target_params)}")
        if world_size > 1:
            print(f"DDP: {world_size} GPUs, effective_batch={cfg.batch_size * cfg.grad_accum_steps * world_size}")
        for n in target_names:
            print(f"  {n}")

    # --- SVD baseline (before locking) ---
    svd_csv = os.path.join(cfg.results_dir, "svd_stats.csv")
    pre_stats = compute_svd_stats(target_names, target_params, top_k=max(cfg.top_k, 3))
    if is_main:
        log_svd_stats(pre_stats, "before_locking", svd_csv)
        print("\nSVD stats BEFORE locking:")
        for s in pre_stats:
            print(f"  {s['name']:60s}  σ1={s['sigma_1']:.4f}  σ2={s['sigma_2']:.4f}  κ={s['condition_number']:.1f}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=cfg.lock_lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # --- Training loop ---
    history = []
    optimizer.zero_grad(set_to_none=True)

    if is_main:
        print(f"\nStarting SpecDef locking for {cfg.lock_steps} steps...")
        print(f"  alpha: {cfg.alpha_start} -> {cfg.alpha_end} (linear)")
        print(f"  top_k={cfg.top_k}, lr={cfg.lock_lr}, seq_len={cfg.seq_len}")
        print(f"  grad_accum={cfg.grad_accum_steps}, effective_batch={cfg.batch_size * cfg.grad_accum_steps * world_size}")

    no_sync = getattr(model, "no_sync", nullcontext)
    trainable_list = [p for p in raw_model.parameters() if p.requires_grad]

    for step in tqdm(range(cfg.lock_steps), disable=not is_main):
        model.train()
        alpha = get_alpha(step, cfg.lock_steps, cfg.alpha_start, cfg.alpha_end)

        # Gradient accumulation — skip all-reduce until last micro-step
        for accum_idx in range(cfg.grad_accum_steps):
            ctx = no_sync() if accum_idx < cfg.grad_accum_steps - 1 else nullcontext()
            with ctx:
                batch = build_batch(tokenizer, train_seqs, cfg.batch_size, cfg.seq_len, cfg.device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = model(batch)
                    retain_loss = causal_lm_loss(logits, batch) / cfg.grad_accum_steps

                spec_term = spectral_term_topk(target_params, top_k=cfg.top_k)
                lock_loss = alpha * retain_loss - (1.0 - alpha) * spec_term / cfg.grad_accum_steps

                if use_scaler:
                    scaler.scale(lock_loss).backward()
                else:
                    lock_loss.backward()

        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_list, cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(trainable_list, cfg.max_grad_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        # --- Logging (rank 0 only) ---
        if (step % cfg.val_every == 0 or step == cfg.lock_steps - 1) and is_main:
            val_loss, val_ppl, val_acc = evaluate(
                model, tokenizer, val_seqs, cfg.device,
                cfg.val_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
            )

            # Recompute spectral term for logging (detached)
            with torch.no_grad():
                spec_log = spectral_term_topk(target_params, cfg.top_k).item()

            record = {
                "step": step,
                "alpha": round(alpha, 4),
                "retain_loss": round(retain_loss.detach().float().item() * cfg.grad_accum_steps, 4),
                "spectral_term": round(spec_log, 4),
                "val_retain_loss": round(val_loss, 4),
                "val_retain_ppl": round(val_ppl, 4),
                "val_retain_acc": round(val_acc, 4),
            }
            history.append(record)
            print(
                f"Step {step:04d} | α={record['alpha']:.2f} | "
                f"retain={record['retain_loss']:.4f} | "
                f"spec={record['spectral_term']:.4f} | "
                f"val_loss={record['val_retain_loss']:.4f} | "
                f"val_acc={record['val_retain_acc']:.4f}"
            )

    # --- Post-training (rank 0 only) ---
    if is_main:
        post_stats = compute_svd_stats(target_names, target_params, top_k=max(cfg.top_k, 3))
        log_svd_stats(post_stats, "after_locking", svd_csv)
        print("\nSVD stats AFTER locking:")
        for s in post_stats:
            print(f"  {s['name']:60s}  σ1={s['sigma_1']:.4f}  σ2={s['sigma_2']:.4f}  κ={s['condition_number']:.1f}")

        print("\nSpectral inflation summary:")
        for pre, post in zip(pre_stats, post_stats):
            ratio = post["sigma_1"] / pre["sigma_1"] if pre["sigma_1"] > 0 else float("inf")
            print(f"  {pre['name']:60s}  σ1: {pre['sigma_1']:.4f} -> {post['sigma_1']:.4f}  ({ratio:.2f}x)")

        save_history_csv(history, os.path.join(cfg.results_dir, "lock_metrics.csv"))
        _save_plot(history, os.path.join(cfg.results_dir, "lock_curve.png"))
        _save_summary(cfg, total, trainable, target_names, history, pre_stats, post_stats)

        if cfg.save_checkpoint:
            ckpt_path = os.path.join(cfg.results_dir, "model_locked.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
            print(f"Saved locked checkpoint: {ckpt_path}")

        print("Done.")

    cleanup_ddp()


# ===========================================================================
# Plotting / reporting helpers
# ===========================================================================
def _save_plot(history, path):
    steps = [h["step"] for h in history]
    plt.figure(figsize=(10, 6))
    plt.plot(steps, [h["retain_loss"] for h in history], "o-", label="Retain loss")
    plt.plot(steps, [h["val_retain_loss"] for h in history], "s-", label="Val retain loss")
    plt.plot(steps, [h["spectral_term"] for h in history], "^-", label="Spectral term (mean)")
    plt.plot(steps, [h["alpha"] for h in history], "x--", label="Alpha", alpha=0.5)
    plt.xlabel("Lock step")
    plt.ylabel("Value")
    plt.title("SpecDef Locking — Evo Hyena blocks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_summary(cfg, total, trainable, target_names, history, pre_stats, post_stats):
    path = os.path.join(cfg.results_dir, "lock_summary.txt")
    with open(path, "w") as f:
        f.write("SpecDef Locking Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model:            {cfg.model_name}\n")
        f.write(f"Retain data:      {cfg.retain_data_path}\n")
        f.write(f"Target blocks:    {sorted(cfg.target_blocks)}\n")
        f.write(f"Lock steps:       {cfg.lock_steps}\n")
        f.write(f"LR:               {cfg.lock_lr}\n")
        f.write(f"Alpha schedule:   {cfg.alpha_start} -> {cfg.alpha_end}\n")
        f.write(f"Top-k:            {cfg.top_k}\n")
        f.write(f"Seq len:          {cfg.seq_len}\n")
        f.write(f"Batch size:       {cfg.batch_size}\n")
        f.write(f"Grad accum:       {cfg.grad_accum_steps}\n")
        f.write(f"Effective batch:  {cfg.batch_size * cfg.grad_accum_steps}\n")
        f.write(f"Max grad norm:    {cfg.max_grad_norm}\n\n")
        f.write(f"Total params:     {total:,}\n")
        f.write(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)\n")
        f.write(f"Target matrices:  {len(target_names)}\n\n")

        f.write("Layer patterns:\n")
        for pat in cfg.target_layer_patterns:
            f.write(f"  {pat}\n")
        f.write("\nTarget matrices:\n")
        for n in target_names:
            f.write(f"  {n}\n")

        f.write("\n\nSpectral inflation:\n")
        f.write(f"{'Name':60s}  {'Before':>10s}  {'After':>10s}  {'Ratio':>8s}\n")
        f.write("-" * 92 + "\n")
        for pre, post in zip(pre_stats, post_stats):
            ratio = post["sigma_1"] / pre["sigma_1"] if pre["sigma_1"] > 0 else float("inf")
            f.write(f"{pre['name']:60s}  {pre['sigma_1']:10.4f}  {post['sigma_1']:10.4f}  {ratio:8.2f}x\n")

        if history:
            final = history[-1]
            f.write(f"\n\nFinal metrics (step {final['step']}):\n")
            f.write(f"  Retain loss:     {final['retain_loss']}\n")
            f.write(f"  Spectral term:   {final['spectral_term']}\n")
            f.write(f"  Val retain loss: {final['val_retain_loss']}\n")
            f.write(f"  Val retain ppl:  {final['val_retain_ppl']}\n")
            f.write(f"  Val retain acc:  {final['val_retain_acc']}\n")

    print(f"Saved summary: {path}")


if __name__ == "__main__":
    main()
