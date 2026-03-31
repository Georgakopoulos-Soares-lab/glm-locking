"""SpecDef locking for Evo — inflate top-k singular values of all linear layers
in target Hyena blocks while preserving model utility via a retain loss.

Usage:
    # Single GPU
    python scripts/lock.py

    # Multi-GPU (torchrun handles LOCAL_RANK / WORLD_SIZE)
    torchrun --nproc_per_node=4 scripts/lock.py
"""

import contextlib
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers.optimization import Adafactor

import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    LockConfig,
    set_seed,
    get_amp_settings,
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
    add_spectral_grad_inplace,
    get_alpha,
    load_evo_model,
    maybe_enable_gradient_checkpointing,
    save_history_csv,
    count_params,
    count_trainable,
    count_params_by_name,
)

# ===========================================================================
# Config — edit here or override via a config file
# ===========================================================================
CONFIG = LockConfig(
    run_name="lock_v9_topk4",
    results_dir="results/lock_v9_topk4",

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
    top_k=4,
    batch_size=2,       # gradient checkpointing frees activation memory; 2 is safe on A100 40GB
    seq_len=512,
    grad_accum_steps=1,
    val_every=50,
    val_batches=8,
    max_grad_norm=1.0,

    target_blocks=set(range(32)),
    target_layer_patterns=(
        ".projections.weight",
        ".out_filter_dense.weight",
        ".mlp.l1.weight",
        ".mlp.l2.weight",
        ".mlp.l3.weight",
    ),

    save_checkpoint=True,
    use_gradient_checkpointing=True,
)


# ===========================================================================
# Distributed helpers
# ===========================================================================

def _maybe_relaunch_torchrun():
    """If not already under torchrun and multiple GPUs are visible, re-exec via torchrun.

    This lets you run `python scripts/lock.py` and automatically get DDP on all
    available GPUs without remembering to use torchrun manually.
    """
    if "LOCAL_RANK" in os.environ:
        return  # Already launched by torchrun — nothing to do

    import subprocess
    n = torch.cuda.device_count()
    if n <= 1:
        return  # Single GPU or no GPU — run normally

    print(f"[auto-launch] {n} GPUs detected. Re-launching with torchrun --nproc_per_node={n}")
    cmd = [
        "torchrun",
        f"--nproc_per_node={n}",
        "--standalone",
    ] + sys.argv  # sys.argv[0] is this script
    sys.exit(subprocess.call(cmd))


def _setup_dist():
    """Initialize torch.distributed if LOCAL_RANK is set (torchrun / srun).

    Returns:
        rank (int)       – global rank; 0 for non-distributed runs
        world_size (int) – total number of processes; 1 for non-distributed
        local_rank (int) – GPU index on this node; 0 for non-distributed
        is_ddp (bool)    – True when running under DDP
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0, False

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


# ===========================================================================
# Main
# ===========================================================================
def main():
    # Auto-relaunch via torchrun if multiple GPUs are available and we weren't
    # already started by torchrun. No-op if LOCAL_RANK is already set.
    _maybe_relaunch_torchrun()

    # --- Distributed setup (no-op on single GPU) ---
    rank, world_size, local_rank, is_ddp = _setup_dist()
    is_rank0 = (rank == 0)

    cfg = CONFIG
    if is_ddp:
        # Each rank uses its own GPU; override the device string from config
        cfg.device = f"cuda:{local_rank}"

    if is_rank0:
        os.makedirs(cfg.results_dir, exist_ok=True)
    set_seed(cfg.seed + rank)   # different seed per rank → different data samples
    amp_dtype, use_scaler = get_amp_settings()

    # --- Data ---
    if is_rank0:
        print("Loading retain data...")
    sequences = load_sequences(cfg.retain_data_path, cfg.min_seq_len)
    train_seqs, val_seqs = split_sequences(sequences, cfg.train_fraction)
    if is_rank0:
        print(f"  {len(sequences)} sequences -> {len(train_seqs)} train / {len(val_seqs)} val")

    # --- Model ---
    model, tokenizer = load_evo_model(cfg.model_name, cfg.device)
    maybe_enable_gradient_checkpointing(model, cfg.use_gradient_checkpointing)

    # --- Targets (before DDP wrapping so we get the real param tensors) ---
    target_names, target_params = get_lock_targets(
        model, cfg.target_blocks, cfg.target_layer_patterns
    )
    if not target_params:
        raise RuntimeError("No target parameters matched. Check target_layer_patterns.")

    target_names_set = set(target_names)
    freeze_all_except(model, target_names_set)

    if is_rank0:
        total = count_params(model)
        trainable = count_trainable(model)
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
        print(f"Target matrices:  {len(target_params)}")
        for n in target_names:
            print(f"  {n}")
    else:
        total = count_params(model)
        trainable = count_trainable(model)

    # --- Wrap with DDP after selecting targets ---
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # --- SVD baseline (before locking) — rank 0 only ---
    if is_rank0:
        svd_csv = os.path.join(cfg.results_dir, "svd_stats.csv")
        pre_stats = compute_svd_stats(target_names, target_params, top_k=max(cfg.top_k, 3))
        log_svd_stats(pre_stats, "before_locking", svd_csv)
        print("\nSVD stats BEFORE locking:")
        for s in pre_stats:
            print(f"  {s['name']:60s}  σ1={s['sigma_1']:.4f}  σ2={s['sigma_2']:.4f}  κ={s['condition_number']:.1f}")
    if is_ddp:
        dist.barrier()  # all ranks wait for rank 0 to finish SVD stats before training starts

    # --- Optimizer (on the trainable params of the raw model, pre-DDP refs still valid) ---
    # AdamW would need 23 GiB of optimizer state for 154 bf16 matrices — doesn't fit on 40 GB GPU
    # alongside the 38 GB model.  Adafactor stores only row+col vectors (O(n+m) per matrix),
    # reducing total optimizer state from ~23 GiB to ~3 MB.
    trainable_param_list = [p for p in target_params if p.requires_grad]
    optimizer = Adafactor(
        trainable_param_list,
        lr=cfg.lock_lr,
        scale_parameter=False,   # use explicit lr, not rms(param)-scaled lr
        relative_step=False,     # use fixed lr provided above
        clip_threshold=1.0,      # Adafactor's internal RMS gradient clipping
        weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # --- Training loop ---
    history = []
    optimizer.zero_grad(set_to_none=True)

    if is_rank0:
        print(f"\nStarting SpecDef locking for {cfg.lock_steps} steps...")
        print(f"  alpha: {cfg.alpha_start} -> {cfg.alpha_end} (linear)")
        print(f"  top_k={cfg.top_k}, lr={cfg.lock_lr}, seq_len={cfg.seq_len}")
        print(f"  grad_accum={cfg.grad_accum_steps}, effective_batch={cfg.batch_size * cfg.grad_accum_steps * world_size}")

    # Helper: suppress DDP gradient sync for gradient accumulation (no-op on single GPU)
    no_sync = model.no_sync if is_ddp else contextlib.nullcontext

    for step in tqdm(range(cfg.lock_steps), disable=not is_rank0):
        model.train()
        alpha = get_alpha(step, cfg.lock_steps, cfg.alpha_start, cfg.alpha_end)

        # ------------------------------------------------------------------
        # Spectral gradient — analytical, GPU sequential, no autograd:
        # d(spec_loss)/d(A_i) = -(1-alpha) / N / k * U[:,:k] @ V[:,:k].T
        # where spec_loss = -(1-alpha) * mean_i(mean_topk_SV(A_i))
        #
        # DDP: since spec_grad is identical on every rank, DDP's gradient
        # all_reduce (averaging) leaves it unchanged — no need to divide by
        # world_size.  This also fixes a bug where the old code made the
        # spectral term world_size× too weak in multi-GPU mode.
        # ------------------------------------------------------------------
        spec_sv_mean = add_spectral_grad_inplace(
            target_params, top_k=cfg.top_k, spec_loss_coeff=-(1.0 - alpha)
        )

        retain_loss_scalar = 0.0  # accumulated for logging

        with no_sync():
            # All retain accum steps except the last stay inside no_sync
            for accum_i in range(cfg.grad_accum_steps - 1):
                batch = build_batch(tokenizer, train_seqs, cfg.batch_size, cfg.seq_len, cfg.device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = model(batch)
                    retain_loss = causal_lm_loss(logits, batch) / cfg.grad_accum_steps
                retain_loss_scalar += retain_loss.detach().float().item()
                if use_scaler:
                    scaler.scale(alpha * retain_loss).backward()
                else:
                    (alpha * retain_loss).backward()

        # Last retain step — exits no_sync, triggers DDP all_reduce for accumulated grads
        batch = build_batch(tokenizer, train_seqs, cfg.batch_size, cfg.seq_len, cfg.device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = model(batch)
            retain_loss = causal_lm_loss(logits, batch) / cfg.grad_accum_steps
        retain_loss_scalar += retain_loss.detach().float().item()
        if use_scaler:
            scaler.scale(alpha * retain_loss).backward()
        else:
            (alpha * retain_loss).backward()

        # --- Optimizer step ---
        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(target_params, cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(target_params, cfg.max_grad_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        # --- Logging (rank 0 only) ---
        if is_rank0 and (step % cfg.val_every == 0 or step == cfg.lock_steps - 1):
            # evaluate() uses the raw DDP-wrapped model; .eval() works on both
            raw_model = model.module if is_ddp else model
            val_loss, val_ppl, val_acc = evaluate(
                raw_model, tokenizer, val_seqs, cfg.device,
                cfg.val_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
            )

            with torch.no_grad():
                spec_log = spectral_term_topk(target_params, cfg.top_k).item()

            record = {
                "step": step,
                "alpha": round(alpha, 4),
                "retain_loss": round(retain_loss_scalar * cfg.grad_accum_steps, 4),
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

    # --- SVD after locking (rank 0 only) ---
    if is_rank0:
        post_stats = compute_svd_stats(target_names, target_params, top_k=max(cfg.top_k, 3))
        log_svd_stats(post_stats, "after_locking", svd_csv)
        print("\nSVD stats AFTER locking:")
        for s in post_stats:
            print(f"  {s['name']:60s}  σ1={s['sigma_1']:.4f}  σ2={s['sigma_2']:.4f}  κ={s['condition_number']:.1f}")

    # --- Print inflation summary + Save (rank 0 only) ---
    if is_rank0:
        print("\nSpectral inflation summary:")
        for pre, post in zip(pre_stats, post_stats):
            ratio = post["sigma_1"] / pre["sigma_1"] if pre["sigma_1"] > 0 else float("inf")
            print(f"  {pre['name']:60s}  σ1: {pre['sigma_1']:.4f} -> {post['sigma_1']:.4f}  ({ratio:.2f}x)")

        save_history_csv(history, os.path.join(cfg.results_dir, "lock_metrics.csv"))
        _save_plot(history, os.path.join(cfg.results_dir, "lock_curve.png"))
        _save_summary(cfg, total, trainable, target_names, history, pre_stats, post_stats)

        if cfg.save_checkpoint:
            raw_model = model.module if is_ddp else model
            ckpt_path = os.path.join(cfg.results_dir, "model_locked.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
            print(f"Saved locked checkpoint: {ckpt_path}")

    if is_rank0:
        print("Done.")
    if is_ddp:
        dist.destroy_process_group()


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
