"""Fine-tune (attack) Evo blocks — runs with locked or unlocked initialization.

This is the "attacker" step: tries to repurpose the model for a new task.
Compare locked vs unlocked checkpoints to measure lock effectiveness.

Usage:
    python scripts/finetune.py
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    FinetuneConfig,
    set_seed,
    get_amp_settings,
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
# Config — edit or swap between locked/unlocked runs
# ===========================================================================
CONFIG = FinetuneConfig(
    run_name="ft_attack_locked_v9",
    results_dir="results/ft_attack_locked_v9",

    data_path="data/attack.fasta",
    model_name="evo-1-8k-base",
    device="cuda:0",
    seed=42,
    train_fraction=0.9,
    min_seq_len=512,

    train_steps=5000,
    lr=5e-5,
    batch_size=1,
    seq_len=512,
    grad_accum_steps=1,
    val_every=200,
    eval_batches=8,
    max_grad_norm=1.0,
    optimizer_name="adamw",

    target_blocks=set(range(8)),
    locked_ckpt="results/lock_v9_topk4/model_locked.pt",

    save_checkpoint=True,
    use_gradient_checkpointing=True,
)


def main():
    cfg = CONFIG
    os.makedirs(cfg.results_dir, exist_ok=True)
    set_seed(cfg.seed)
    amp_dtype, use_scaler = get_amp_settings()

    # --- Data ---
    print("Loading attack data...")
    sequences = load_sequences(cfg.data_path, cfg.min_seq_len)
    train_seqs, val_seqs = split_sequences(sequences, cfg.train_fraction)
    total_train_nt = sum(len(s) for s in train_seqs)
    print(f"  {len(sequences)} sequences -> {len(train_seqs)} train / {len(val_seqs)} val")

    # --- Model ---
    model, tokenizer = load_evo_model(cfg.model_name, cfg.device)
    maybe_load_locked_checkpoint(model, cfg.locked_ckpt)
    maybe_enable_gradient_checkpointing(model, cfg.use_gradient_checkpointing)

    # --- Targets (all params in blocks 0-7) ---
    target_names = get_block_params(model, cfg.target_blocks)
    target_names_set = set(target_names)
    freeze_all_except(model, target_names_set)

    total = count_params(model)
    trainable = count_trainable(model)
    targeted = count_params_by_name(model, target_names)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"Targeted params:  {targeted:,}")

    # --- Optimizer ---
    optimizer = build_optimizer(model, cfg.optimizer_name, cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    tokens_seen = cfg.train_steps * cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps
    coverage = tokens_seen / total_train_nt if total_train_nt > 0 else 0.0
    print(f"Tokens to see: {tokens_seen:,} ({100 * coverage:.2f}% of train data)")

    # --- Training loop ---
    history = []
    best_val_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)

    print(f"\nFine-tuning for {cfg.train_steps} steps (locked_ckpt={'yes' if cfg.locked_ckpt else 'no'})...")

    for step in tqdm(range(cfg.train_steps)):
        model.train()

        for _ in range(cfg.grad_accum_steps):
            batch = build_batch(tokenizer, train_seqs, cfg.batch_size, cfg.seq_len, cfg.device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, _ = model(batch)
                loss = causal_lm_loss(logits, batch) / cfg.grad_accum_steps
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        trainable_list = [p for p in model.parameters() if p.requires_grad]
        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_list, cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(trainable_list, cfg.max_grad_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.val_every == 0 or step == cfg.train_steps - 1:
            train_loss, train_ppl, train_acc = evaluate(
                model, tokenizer, train_seqs, cfg.device,
                cfg.eval_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
            )
            val_loss, val_ppl, val_acc = evaluate(
                model, tokenizer, val_seqs, cfg.device,
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
            print(
                f"Step {step:05d} | "
                f"train_loss={record['train_loss']:.4f} | "
                f"val_loss={record['val_loss']:.4f} | "
                f"val_acc={record['val_acc']:.4f}{marker}"
            )

    # --- Save ---
    save_history_csv(history, os.path.join(cfg.results_dir, "metrics.csv"))
    _save_summary(cfg, total, trainable, targeted, target_names, history,
                  len(train_seqs), len(val_seqs), total_train_nt, tokens_seen, coverage)

    if cfg.save_checkpoint:
        ckpt_path = os.path.join(cfg.results_dir, "model_finetuned.pt")
        torch.save(model.state_dict(), ckpt_path)
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
