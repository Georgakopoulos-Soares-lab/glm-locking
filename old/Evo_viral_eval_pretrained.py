import os

from Evo_viral_ft_utils import (
    RunConfig,
    set_seed,
    get_amp_settings,
    load_sequences,
    split_sequences,
    load_evo_model,
    maybe_enable_gradient_checkpointing,
    count_total_params,
    estimate_total_train_nt,
    save_history_csv,
    evaluate,
)


RUN_NAME = "eval_pretrained_pilot758_len128_clean"
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
    train_steps=0,
    lr=0.0,
    batch_size=1,
    seq_len=128,
    grad_accum_steps=1,
    val_every=0,
    eval_batches=16,
    save_checkpoint=False,
    use_gradient_checkpointing=True,
    max_grad_norm=1.0,
    optimizer_name="adamw",
    target_blocks={0, 1, 2, 3, 4, 5, 6, 7},
    locked_ckpt=None,
)


def save_summary(
    out_txt,
    config,
    amp_dtype,
    total_params,
    n_train,
    n_val,
    train_lengths,
    val_lengths,
    total_train_nt,
    train_metrics,
    val_metrics,
):
    train_loss, train_ppl, train_acc = train_metrics
    val_loss, val_ppl, val_acc = val_metrics

    with open(out_txt, "w") as f:
        f.write("Pretrained Evo evaluation summary\n")
        f.write("================================\n\n")
        f.write(f"Run name: {config.run_name}\n")
        f.write(f"Model: {config.model_name}\n")
        f.write(f"Data path: {config.data_path}\n")
        f.write(f"Device: {config.device}\n")
        f.write(f"Train sequences: {n_train}\n")
        f.write(f"Val sequences: {n_val}\n")
        f.write(f"Train length range: {min(train_lengths)} - {max(train_lengths)}\n")
        f.write(f"Val length range: {min(val_lengths)} - {max(val_lengths)}\n")
        f.write(f"Seq len: {config.seq_len}\n")
        f.write(f"Eval batches: {config.eval_batches}\n")
        f.write(f"Gradient checkpointing requested: {config.use_gradient_checkpointing}\n")
        f.write(f"AMP dtype: {amp_dtype}\n\n")
        f.write(f"Total params: {total_params:,}\n")
        f.write(f"Total train nucleotides: {total_train_nt:,}\n\n")
        f.write(f"Train loss: {train_loss:.6f}\n")
        f.write(f"Train ppl: {train_ppl:.6f}\n")
        f.write(f"Train acc: {train_acc:.6f}\n\n")
        f.write(f"Val loss: {val_loss:.6f}\n")
        f.write(f"Val ppl: {val_ppl:.6f}\n")
        f.write(f"Val acc: {val_acc:.6f}\n")


def main():
    os.makedirs(CONFIG.results_dir, exist_ok=True)

    set_seed(CONFIG.seed)
    amp_dtype, _ = get_amp_settings()

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
    maybe_enable_gradient_checkpointing(model, CONFIG.use_gradient_checkpointing)

    total_params = count_total_params(model)

    print("Evaluating pretrained model on TRAIN split...")
    train_metrics = evaluate(
        model=model,
        tokenizer=tokenizer,
        seq_pool=train_seqs,
        device=CONFIG.device,
        eval_batches=CONFIG.eval_batches,
        batch_size=CONFIG.batch_size,
        seq_len=CONFIG.seq_len,
        amp_dtype=amp_dtype,
    )

    print("Evaluating pretrained model on VAL split...")
    val_metrics = evaluate(
        model=model,
        tokenizer=tokenizer,
        seq_pool=val_seqs,
        device=CONFIG.device,
        eval_batches=CONFIG.eval_batches,
        batch_size=CONFIG.batch_size,
        seq_len=CONFIG.seq_len,
        amp_dtype=amp_dtype,
    )

    train_loss, train_ppl, train_acc = train_metrics
    val_loss, val_ppl, val_acc = val_metrics

    print(
        f"Pretrained eval TRAIN | loss={train_loss:.4f} | ppl={train_ppl:.4f} | acc={train_acc:.4f}"
    )
    print(
        f"Pretrained eval VAL   | loss={val_loss:.4f} | ppl={val_ppl:.4f} | acc={val_acc:.4f}"
    )

    history = [
        {
            "split": "train",
            "loss": float(train_loss),
            "perplexity": float(train_ppl),
            "next_token_acc": float(train_acc),
            "run_name": CONFIG.run_name,
        },
        {
            "split": "val",
            "loss": float(val_loss),
            "perplexity": float(val_ppl),
            "next_token_acc": float(val_acc),
            "run_name": CONFIG.run_name,
        },
    ]

    csv_path = os.path.join(CONFIG.results_dir, "metrics.csv")
    txt_path = os.path.join(CONFIG.results_dir, "run_summary.txt")

    save_history_csv(history, csv_path)
    save_summary(
        txt_path,
        CONFIG,
        amp_dtype,
        total_params,
        len(train_seqs),
        len(val_seqs),
        train_lengths,
        val_lengths,
        total_train_nt,
        train_metrics,
        val_metrics,
    )

    print(f"\nSaved metrics to: {csv_path}")
    print(f"Saved summary to: {txt_path}")


if __name__ == "__main__":
    main()
