import os

import torch
from tqdm import tqdm

from Evo_viral_ft_utils import (
    RunConfig,
    set_seed,
    get_amp_settings,
    load_sequences,
    split_sequences,
    load_evo_model,
    maybe_load_locked_checkpoint,
    maybe_enable_gradient_checkpointing,
    get_block_target_names,
    freeze_all_except_targets,
    count_total_params,
    count_params_by_name,
    fraction_str,
    estimate_tokens_seen,
    estimate_total_train_nt,
    estimate_coverage_ratio,
    build_optimizer,
    save_history_csv,
    evaluate,
    causal_lm_loss,
    build_batch_from_sequences,
)


#RUN_NAME = "ft_blocks_locked_v7_pilot758_len128_clean"
RUN_NAME = "ft_blocks_locked_v7_pilot758_len128_clean_2k"
RESULTS_DIR = f"results/{RUN_NAME}"

CONFIG = RunConfig(
    run_name=RUN_NAME,
    results_dir=RESULTS_DIR,
    model_name="evo-1-8k-base",
    device="cuda:0",
    data_path="viral_pilot_raw/data/viral_pilot_balanced_min8k.fasta",
    seed=42,
    train_fraction=0.9,
    min_seq_len=128,
    train_steps=2000,
    lr=1e-5,
    batch_size=1,
    seq_len=128,
    grad_accum_steps=1,
    val_every=50,
    eval_batches=16,
    save_checkpoint=True,
    use_gradient_checkpointing=True,
    max_grad_norm=1.0,
    optimizer_name="adamw",
    target_blocks={0, 1, 2, 3, 4, 5, 6, 7},
    locked_ckpt="results/evo_lock_projections_v7/model_locked_projections.pt",
)


def save_summary(
    out_txt,
    config,
    amp_dtype,
    total_params,
    trainable_params,
    targeted_params,
    n_train,
    n_val,
    train_lengths,
    val_lengths,
    total_train_nt,
    tokens_seen,
    coverage_ratio,
    target_names,
    history,
):
    best_val_loss_record = min(history, key=lambda x: x["val_loss"])
    best_val_ppl_record = min(history, key=lambda x: x["val_perplexity"])
    best_val_acc_record = max(history, key=lambda x: x["val_next_token_acc"])
    final_record = history[-1]

    with open(out_txt, "w") as f:
        f.write("FASTA broader block fine-tuning summary\n")
        f.write("=======================================\n\n")
        f.write(f"Run name: {config.run_name}\n")
        f.write(f"Model: {config.model_name}\n")
        f.write(f"Data path: {config.data_path}\n")
        f.write(f"Locked checkpoint: {config.locked_ckpt}\n")
        f.write(f"Target blocks: {sorted(config.target_blocks)}\n")
        f.write(f"Device: {config.device}\n")
        f.write(f"Train sequences: {n_train}\n")
        f.write(f"Val sequences: {n_val}\n")
        f.write(f"Train length range: {min(train_lengths)} - {max(train_lengths)}\n")
        f.write(f"Val length range: {min(val_lengths)} - {max(val_lengths)}\n")
        f.write(f"Train steps: {config.train_steps}\n")
        f.write(f"Batch size: {config.batch_size}\n")
        f.write(f"Seq len: {config.seq_len}\n")
        f.write(f"Grad accum steps: {config.grad_accum_steps}\n")
        f.write(f"Optimizer: {config.optimizer_name}\n")
        f.write(f"Learning rate: {config.lr}\n")
        f.write(f"Eval batches: {config.eval_batches}\n")
        f.write(f"Gradient checkpointing requested: {config.use_gradient_checkpointing}\n")
        f.write(f"AMP dtype: {amp_dtype}\n\n")

        f.write(f"Total params: {total_params:,}\n")
        f.write(f"Trainable params: {trainable_params:,} ({fraction_str(trainable_params, total_params)})\n")
        f.write(f"Targeted params: {targeted_params:,} ({fraction_str(targeted_params, total_params)})\n")
        f.write(f"Number of trainable parameter tensors selected: {len(target_names)}\n\n")

        f.write(f"Total train nucleotides: {total_train_nt:,}\n")
        f.write(f"Tokens seen in training: {tokens_seen:,}\n")
        f.write(f"Approximate coverage ratio: {coverage_ratio:.6f}\n\n")

        f.write("Best validation metrics\n")
        f.write("-----------------------\n")
        f.write(f"Best val loss step: {best_val_loss_record['step']}\n")
        f.write(f"Best val loss: {best_val_loss_record['val_loss']:.6f}\n")
        f.write(f"Best val ppl step: {best_val_ppl_record['step']}\n")
        f.write(f"Best val ppl: {best_val_ppl_record['val_perplexity']:.6f}\n")
        f.write(f"Best val acc step: {best_val_acc_record['step']}\n")
        f.write(f"Best val acc: {best_val_acc_record['val_next_token_acc']:.6f}\n\n")

        f.write("Final metrics\n")
        f.write("-------------\n")
        f.write(f"Final train loss: {final_record['train_loss']:.6f}\n")
        f.write(f"Final train ppl: {final_record['train_perplexity']:.6f}\n")
        f.write(f"Final train acc: {final_record['train_next_token_acc']:.6f}\n")
        f.write(f"Final val loss: {final_record['val_loss']:.6f}\n")
        f.write(f"Final val ppl: {final_record['val_perplexity']:.6f}\n")
        f.write(f"Final val acc: {final_record['val_next_token_acc']:.6f}\n")


def main():
    os.makedirs(CONFIG.results_dir, exist_ok=True)

    set_seed(CONFIG.seed)
    amp_dtype, use_scaler = get_amp_settings()

    print("Loading sequences...")
    sequences = load_sequences(CONFIG.data_path, CONFIG.min_seq_len)
    train_seqs, val_seqs = split_sequences(sequences, CONFIG.train_fraction)

    train_lengths = [len(x) for x in train_seqs]
    val_lengths = [len(x) for x in val_seqs]
    total_train_nt = estimate_total_train_nt(train_seqs)

    print(f"Loaded {len(sequences)} usable sequences from {CONFIG.data_path}")
    print(f"Train sequences: {len(train_seqs)}")
    print(f"Val sequences: {len(val_seqs)}")
    print(f"Train length range: {min(train_lengths)} - {max(train_lengths)}")
    print(f"Val length range: {min(val_lengths)} - {max(val_lengths)}")

    model, tokenizer = load_evo_model(CONFIG.model_name, CONFIG.device)
    maybe_load_locked_checkpoint(model, CONFIG.locked_ckpt)
    maybe_enable_gradient_checkpointing(model, CONFIG.use_gradient_checkpointing)

    target_names = get_block_target_names(model, CONFIG.target_blocks)
    target_names_set = set(target_names)

    print(f"Target block subset: {sorted(CONFIG.target_blocks)}")
    print(f"Found {len(target_names)} trainable parameter tensors inside target blocks")
    for name in target_names[:50]:
        print(f"  {name}")
    if len(target_names) > 50:
        print(f"  ... and {len(target_names) - 50} more")

    if len(target_names) == 0:
        raise RuntimeError("No parameters selected inside target blocks.")

    freeze_all_except_targets(model, target_names_set)

    total_params = count_total_params(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    targeted_params = count_params_by_name(model, target_names)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({fraction_str(trainable_params, total_params)})")
    print(f"Targeted params: {targeted_params:,} ({fraction_str(targeted_params, total_params)})")

    optimizer = build_optimizer(model, CONFIG.optimizer_name, CONFIG.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    history = []
    optimizer.zero_grad(set_to_none=True)

    tokens_seen = estimate_tokens_seen(
        train_steps=CONFIG.train_steps,
        batch_size=CONFIG.batch_size,
        seq_len=CONFIG.seq_len,
        grad_accum_steps=CONFIG.grad_accum_steps,
    )
    coverage_ratio = estimate_coverage_ratio(tokens_seen, total_train_nt)

    print(f"Run name: {CONFIG.run_name}")
    print(f"Results dir: {CONFIG.results_dir}")
    print(f"Using AMP dtype: {amp_dtype}")
    print(f"Using GradScaler: {use_scaler}")
    print(f"Training broader block fine-tuning for {CONFIG.train_steps} steps...")
    print(
        f"LR={CONFIG.lr}, BATCH_SIZE={CONFIG.batch_size}, SEQ_LEN={CONFIG.seq_len}, "
        f"VAL_EVERY={CONFIG.val_every}, EVAL_BATCHES={CONFIG.eval_batches}"
    )
    print(f"Tokens seen in training: {tokens_seen:,}")
    print(f"Approximate coverage ratio: {coverage_ratio:.6f}")

    for step in tqdm(range(CONFIG.train_steps)):
        model.train()

        for _ in range(CONFIG.grad_accum_steps):
            batch = build_batch_from_sequences(
                tokenizer, train_seqs, CONFIG.batch_size, CONFIG.seq_len, CONFIG.device
            )

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, _ = model(batch)
                loss = causal_lm_loss(logits, batch)
                loss = loss / CONFIG.grad_accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        trainable_param_list = [p for p in model.parameters() if p.requires_grad]

        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_param_list, CONFIG.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(trainable_param_list, CONFIG.max_grad_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if step % CONFIG.val_every == 0 or step == CONFIG.train_steps - 1:
            train_loss, train_ppl, train_acc = evaluate(
                model=model,
                tokenizer=tokenizer,
                seq_pool=train_seqs,
                device=CONFIG.device,
                eval_batches=CONFIG.eval_batches,
                batch_size=CONFIG.batch_size,
                seq_len=CONFIG.seq_len,
                amp_dtype=amp_dtype,
            )

            val_loss, val_ppl, val_acc = evaluate(
                model=model,
                tokenizer=tokenizer,
                seq_pool=val_seqs,
                device=CONFIG.device,
                eval_batches=CONFIG.eval_batches,
                batch_size=CONFIG.batch_size,
                seq_len=CONFIG.seq_len,
                amp_dtype=amp_dtype,
            )

            record = {
                "step": step,
                "train_loss": float(train_loss),
                "train_perplexity": float(train_ppl),
                "train_next_token_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_perplexity": float(val_ppl),
                "val_next_token_acc": float(val_acc),
                "run_name": CONFIG.run_name,
                "locked_ckpt": CONFIG.locked_ckpt,
                "optimizer": CONFIG.optimizer_name,
                "lr": CONFIG.lr,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "trainable_fraction_percent": 100.0 * trainable_params / total_params,
                "targeted_params": targeted_params,
                "targeted_fraction_percent": 100.0 * targeted_params / total_params,
                "tokens_seen": tokens_seen,
                "coverage_ratio": coverage_ratio,
            }
            history.append(record)

            print(
                f"Step {step:03d} | "
                f"train_loss={record['train_loss']:.4f} | "
                f"train_ppl={record['train_perplexity']:.4f} | "
                f"train_acc={record['train_next_token_acc']:.4f} | "
                f"val_loss={record['val_loss']:.4f} | "
                f"val_ppl={record['val_perplexity']:.4f} | "
                f"val_acc={record['val_next_token_acc']:.4f}"
            )

    csv_path = os.path.join(CONFIG.results_dir, "metrics.csv")
    txt_path = os.path.join(CONFIG.results_dir, "run_summary.txt")
    ckpt_path = os.path.join(CONFIG.results_dir, "model_finetuned_blocks.pt")

    save_history_csv(history, csv_path)
    save_summary(
        txt_path,
        CONFIG,
        amp_dtype,
        total_params,
        trainable_params,
        targeted_params,
        len(train_seqs),
        len(val_seqs),
        train_lengths,
        val_lengths,
        total_train_nt,
        tokens_seen,
        coverage_ratio,
        target_names,
        history,
    )

    if CONFIG.save_checkpoint:
        torch.save(model.state_dict(), ckpt_path)

    print(f"\nSaved metrics to: {csv_path}")
    print(f"Saved summary to: {txt_path}")
    if CONFIG.save_checkpoint:
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
