"""Evaluate the pretrained Evo model (no fine-tuning) as a baseline.

Usage:
    python scripts/eval_pretrained.py
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    EvalConfig,
    set_seed,
    get_amp_settings,
    load_sequences,
    split_sequences,
    evaluate,
    load_evo_model,
    maybe_enable_gradient_checkpointing,
    save_history_csv,
    count_params,
)

# ===========================================================================
# Config
# ===========================================================================
CONFIG = EvalConfig(
    run_name="eval_pretrained_v10",
    results_dir="results/eval_pretrained_v10",
    data_path="data/attack.fasta",
    model_name="evo-1-8k-base",
    device="cuda:0",
    seed=42,
    train_fraction=0.9,
    min_seq_len=1024,
    seq_len=1024,
    eval_batches=16,
    batch_size=1,
    use_gradient_checkpointing=True,
)


def main():
    cfg = CONFIG
    os.makedirs(cfg.results_dir, exist_ok=True)
    set_seed(cfg.seed)
    amp_dtype, _ = get_amp_settings()

    print("Loading data...")
    sequences = load_sequences(cfg.data_path, cfg.min_seq_len)
    train_seqs, val_seqs = split_sequences(sequences, cfg.train_fraction)
    print(f"  {len(sequences)} sequences -> {len(train_seqs)} train / {len(val_seqs)} val")

    model, tokenizer = load_evo_model(cfg.model_name, cfg.device)
    maybe_enable_gradient_checkpointing(model, cfg.use_gradient_checkpointing)
    total = count_params(model)

    print("Evaluating on TRAIN split...")
    train_loss, train_ppl, train_acc = evaluate(
        model, tokenizer, train_seqs, cfg.device,
        cfg.eval_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
    )

    print("Evaluating on VAL split...")
    val_loss, val_ppl, val_acc = evaluate(
        model, tokenizer, val_seqs, cfg.device,
        cfg.eval_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
    )

    print(f"TRAIN | loss={train_loss:.4f} | ppl={train_ppl:.4f} | acc={train_acc:.4f}")
    print(f"VAL   | loss={val_loss:.4f}  | ppl={val_ppl:.4f}  | acc={val_acc:.4f}")

    history = [
        {"split": "train", "loss": train_loss, "ppl": train_ppl, "acc": train_acc},
        {"split": "val",   "loss": val_loss,   "ppl": val_ppl,   "acc": val_acc},
    ]
    save_history_csv(history, os.path.join(cfg.results_dir, "metrics.csv"))

    # Summary
    path = os.path.join(cfg.results_dir, "run_summary.txt")
    with open(path, "w") as f:
        f.write("Pretrained Evaluation Baseline\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model:      {cfg.model_name}\n")
        f.write(f"Data:       {cfg.data_path}\n")
        f.write(f"Seq len:    {cfg.seq_len}\n")
        f.write(f"Eval batch: {cfg.eval_batches}\n")
        f.write(f"Total params: {total:,}\n\n")
        f.write(f"Train: loss={train_loss:.6f}  ppl={train_ppl:.6f}  acc={train_acc:.6f}\n")
        f.write(f"Val:   loss={val_loss:.6f}  ppl={val_ppl:.6f}  acc={val_acc:.6f}\n")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
