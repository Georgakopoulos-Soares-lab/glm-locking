"""Fine-tune (attack) Evo blocks — runs with locked or unlocked initialization.

This is the "attacker" step: tries to repurpose the model for a new task.
Compare locked vs unlocked checkpoints to measure lock effectiveness.

Usage:
    # Via YAML config (preferred):
    python scripts/finetune.py --config configs/ft_v10.yaml               # mode from YAML
    python scripts/finetune.py --config configs/ft_v10.yaml --locked      # override: locked only
    python scripts/finetune.py --config configs/ft_v10.yaml --unlocked    # override: unlocked only

    # Via hardcoded defaults:
    python scripts/finetune.py --locked
    python scripts/finetune.py --unlocked
"""

import os
import sys
import argparse
from contextlib import nullcontext

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    FinetuneConfig,
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
    get_block_params,
    freeze_all_except,
    load_evo_model,
    maybe_load_locked_checkpoint,
    maybe_enable_gradient_checkpointing,
    build_optimizer,
    save_history_csv,
    count_params,
    count_trainable,
    count_params_by_name,
)

# ===========================================================================
# Shared hyperparameters (identical for locked and unlocked — fair comparison)
# ===========================================================================
_SHARED = dict(
    data_path="data/attack.fasta",
    model_name="evo-1-8k-base",
    device="cuda:0",
    seed=42,
    train_fraction=0.9,
    min_seq_len=1024,

    train_steps=5000,
    lr=1e-5,
    batch_size=1,
    seq_len=1024,
    grad_accum_steps=4,
    val_every=100,
    eval_batches=16,
    max_grad_norm=1.0,
    optimizer_name="adamw",

    # Attack ALL 32 blocks (must match what was locked)
    target_blocks=set(range(32)),

    save_checkpoint=True,
    use_gradient_checkpointing=True,
)

LOCKED_CONFIG = FinetuneConfig(
    run_name="ft_attack_locked_v10",
    results_dir="results/ft_attack_locked_v10",
    locked_ckpt="results/lock_v10/model_locked.pt",
    **_SHARED,
)

UNLOCKED_CONFIG = FinetuneConfig(
    run_name="ft_attack_unlocked_v10",
    results_dir="results/ft_attack_unlocked_v10",
    locked_ckpt=None,
    **_SHARED,
)


# ===========================================================================
# YAML config loader
# ===========================================================================
def _load_config(path: str, mode: str | None = None) -> list[FinetuneConfig]:
    """Load FinetuneConfig(s) from a YAML file.

    YAML must have a 'mode' field: 'locked', 'unlocked', or 'both'.
    The `mode` argument overrides the YAML field if provided.

    For mode='both', two configs are returned (locked first, unlocked second).
    The locked variant gets run_name + '_locked', the unlocked gets '_unlocked'.
    locked_ckpt is required in the YAML when mode includes locked.
    When mode includes unlocked, locked_ckpt is set to None for that config.
    """
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f)

    yaml_mode = d.pop("mode", "both")
    effective_mode = mode or yaml_mode

    # Auto-derive results_dir from run_name
    base_run_name = d["run_name"]

    if "target_blocks" in d:
        val = d["target_blocks"]
        d["target_blocks"] = set(range(val)) if isinstance(val, int) else set(val)
    # YAML loads scientific notation as string (e.g. '5e-5') — cast to float
    if "lr" in d:
        d["lr"] = float(d["lr"])

    locked_ckpt = d.pop("locked_ckpt", None)

    # Drop keys not in the dataclass
    from dataclasses import fields as _fields
    valid = {f.name for f in _fields(FinetuneConfig)}
    d = {k: v for k, v in d.items() if k in valid}

    configs = []
    if effective_mode in ("locked", "both"):
        if not locked_ckpt:
            raise ValueError(f"mode={effective_mode} requires 'locked_ckpt' in YAML")
        ld = {**d,
              "run_name": f"{base_run_name}_locked",
              "results_dir": f"results/{base_run_name}_locked",
              "locked_ckpt": locked_ckpt}
        configs.append(FinetuneConfig(**ld))
    if effective_mode in ("unlocked", "both"):
        ud = {**d,
              "run_name": f"{base_run_name}_unlocked",
              "results_dir": f"results/{base_run_name}_unlocked",
              "locked_ckpt": None}
        configs.append(FinetuneConfig(**ud))

    if not configs:
        raise ValueError(f"Invalid mode: {effective_mode}. Use 'locked', 'unlocked', or 'both'.")
    return configs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune (attack) Evo")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--locked", action="store_true", help="Attack locked checkpoint")
    group.add_argument("--unlocked", action="store_true", help="Attack pretrained (unlocked) baseline")
    args, _ = parser.parse_known_args()

    # --- DDP setup (once for the whole process) ---
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)

    if args.config:
        mode = "locked" if args.locked else ("unlocked" if args.unlocked else None)
        configs = _load_config(args.config, mode=mode)
    else:
        if not (args.locked or args.unlocked):
            parser.error("Either --config or --locked/--unlocked is required")
        configs = [LOCKED_CONFIG if args.locked else UNLOCKED_CONFIG]

    for i, cfg in enumerate(configs):
        if len(configs) > 1 and is_main:
            print(f"\n{'='*60}")
            print(f" STAGE {i+1}/{len(configs)}: {cfg.run_name}")
            print(f" locked_ckpt={'yes' if cfg.locked_ckpt else 'no'}")
            print(f"{'='*60}\n")
        _run_finetune(cfg, rank, local_rank, world_size)

    cleanup_ddp()


def _run_finetune(cfg: FinetuneConfig, rank: int = 0, local_rank: int = 0, world_size: int = 1):
    is_main = (rank == 0)
    cfg.device = f"cuda:{local_rank}"
    if is_main:
        os.makedirs(cfg.results_dir, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()

    # Deterministic split (same across ranks)
    set_seed(cfg.seed)
    amp_dtype, use_scaler = get_amp_settings()

    # --- Data ---
    if is_main:
        print("Loading attack data...")
    sequences = load_sequences(cfg.data_path, cfg.min_seq_len)
    train_seqs, val_seqs = split_sequences(sequences, cfg.train_fraction)
    total_train_nt = sum(len(s) for s in train_seqs)
    if is_main:
        print(f"  {len(sequences)} sequences -> {len(train_seqs)} train / {len(val_seqs)} val")

    # Re-seed so each rank samples different batches
    set_seed(cfg.seed + rank)

    # --- Model ---
    model, tokenizer = load_evo_model(cfg.model_name, cfg.device)
    maybe_load_locked_checkpoint(model, cfg.locked_ckpt)
    maybe_enable_gradient_checkpointing(model, cfg.use_gradient_checkpointing)

    # --- Targets (all params in target blocks) ---
    target_names = get_block_params(model, cfg.target_blocks)
    target_names_set = set(target_names)
    frozen_count, trainable_count = freeze_all_except(model, target_names_set)

    # --- DDP wrap ---
    model, raw_model = wrap_ddp(model, local_rank)

    total = count_params(raw_model)
    trainable = count_trainable(raw_model)
    targeted = count_params_by_name(raw_model, target_names)
    if is_main:
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
        print(f"Targeted params:  {targeted:,}")
        if world_size > 1:
            print(f"DDP: {world_size} GPUs, effective_batch={cfg.batch_size * cfg.grad_accum_steps * world_size}")

    # --- Optimizer ---
    optimizer = build_optimizer(raw_model, cfg.optimizer_name, cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    tokens_seen = cfg.train_steps * cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps * world_size
    coverage = tokens_seen / total_train_nt if total_train_nt > 0 else 0.0
    if is_main:
        print(f"Tokens to see: {tokens_seen:,} ({100 * coverage:.2f}% of train data)")

    # --- Training loop ---
    history = []
    best_val_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)
    no_sync = getattr(model, "no_sync", nullcontext)
    trainable_list = [p for p in raw_model.parameters() if p.requires_grad]

    if is_main:
        print(f"\nFine-tuning for {cfg.train_steps} steps (locked_ckpt={'yes' if cfg.locked_ckpt else 'no'})...")

    for step in tqdm(range(cfg.train_steps), disable=not is_main):
        model.train()

        for accum_idx in range(cfg.grad_accum_steps):
            ctx = no_sync() if accum_idx < cfg.grad_accum_steps - 1 else nullcontext()
            with ctx:
                batch = build_batch(tokenizer, train_seqs, cfg.batch_size, cfg.seq_len, cfg.device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = model(batch)
                    loss = causal_lm_loss(logits, batch) / cfg.grad_accum_steps
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_list, cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(trainable_list, cfg.max_grad_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (step % cfg.val_every == 0 or step == cfg.train_steps - 1) and is_main:
            train_loss, train_ppl, train_acc = evaluate(
                raw_model, tokenizer, train_seqs, cfg.device,
                cfg.eval_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
            )
            val_loss, val_ppl, val_acc = evaluate(
                raw_model, tokenizer, val_seqs, cfg.device,
                cfg.eval_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
            )

            record = {
                "step": step,
                "train_loss": round(train_loss, 4),
                "train_ppl": round(train_ppl, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 4),
                "val_acc": round(val_acc, 4),
            }
            history.append(record)

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                marker = " *"
                if cfg.save_checkpoint and is_main:
                    best_ckpt = os.path.join(cfg.results_dir, "model_best.pt")
                    torch.save(raw_model.state_dict(), best_ckpt)
            print(
                f"Step {step:05d} | "
                f"train_loss={record['train_loss']:.4f} | "
                f"val_loss={record['val_loss']:.4f} | "
                f"val_acc={record['val_acc']:.4f}{marker}"
            )
        
        # Barrier: ensure all ranks wait for rank 0 to finish validation/checkpointing
        if world_size > 1 and (step % cfg.val_every == 0 or step == cfg.train_steps - 1):
            torch.distributed.barrier()

    # --- Save (rank 0 only) ---
    if is_main:
        save_history_csv(history, os.path.join(cfg.results_dir, "metrics.csv"))
        _save_summary(cfg, total, trainable, targeted, target_names, history,
                      len(train_seqs), len(val_seqs), total_train_nt, tokens_seen, coverage)

        if cfg.save_checkpoint:
            ckpt_path = os.path.join(cfg.results_dir, "model_finetuned.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        print("Done.")


def _save_summary(cfg, total, trainable, targeted, target_names, history,
                  n_train, n_val, total_train_nt, tokens_seen, coverage):
    path = os.path.join(cfg.results_dir, "run_summary.txt")

    best_val = min(history, key=lambda x: x["val_loss"])
    final = history[-1]

    with open(path, "w") as f:
        f.write("Fine-tuning (Attack) Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Run name:       {cfg.run_name}\n")
        f.write(f"Model:          {cfg.model_name}\n")
        f.write(f"Attack data:    {cfg.data_path}\n")
        f.write(f"Locked ckpt:    {cfg.locked_ckpt}\n")
        f.write(f"Target blocks:  {sorted(cfg.target_blocks)}\n")
        f.write(f"Train seqs:     {n_train}\n")
        f.write(f"Val seqs:       {n_val}\n")
        f.write(f"Train steps:    {cfg.train_steps}\n")
        f.write(f"LR:             {cfg.lr}\n")
        f.write(f"Seq len:        {cfg.seq_len}\n")
        f.write(f"Batch size:     {cfg.batch_size}\n")
        f.write(f"Grad accum:     {cfg.grad_accum_steps}\n")
        f.write(f"Optimizer:      {cfg.optimizer_name}\n\n")
        f.write(f"Total params:     {total:,}\n")
        f.write(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)\n")
        f.write(f"Targeted params:  {targeted:,}\n")
        f.write(f"Tokens seen:      {tokens_seen:,}\n")
        f.write(f"Coverage ratio:   {coverage:.4f}\n\n")
        f.write(f"Best val loss:  {best_val['val_loss']:.6f}  (step {best_val['step']})\n")
        f.write(f"Best val acc:   {max(h['val_acc'] for h in history):.6f}\n\n")
        f.write(f"Final train_loss={final['train_loss']:.6f}  val_loss={final['val_loss']:.6f}\n")

    print(f"Saved summary: {path}")


if __name__ == "__main__":
    main()
