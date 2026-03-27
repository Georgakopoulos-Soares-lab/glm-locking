import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim

from evo import Evo


@dataclass
class RunConfig:
    run_name: str
    results_dir: str
    model_name: str
    device: str
    data_path: str
    seed: int
    train_fraction: float
    min_seq_len: int
    train_steps: int
    lr: float
    batch_size: int
    seq_len: int
    grad_accum_steps: int
    val_every: int
    eval_batches: int
    save_checkpoint: bool
    use_gradient_checkpointing: bool
    max_grad_norm: float
    optimizer_name: str
    target_blocks: set
    locked_ckpt: str | None


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_settings():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    bf16_ok = torch.cuda.is_bf16_supported()
    if bf16_ok:
        return torch.bfloat16, False
    return torch.float16, True


def clean_dna(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return "".join(c for c in seq if c in {"A", "C", "G", "T"})


def load_sequences(path: str, min_seq_len: int):
    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA_PATH does not exist: {path}")

    sequences = []

    if path.endswith((".fa", ".fasta", ".fna")):
        current = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current:
                        seq = clean_dna("".join(current))
                        if len(seq) >= min_seq_len:
                            sequences.append(seq)
                        current = []
                else:
                    current.append(line)

        if current:
            seq = clean_dna("".join(current))
            if len(seq) >= min_seq_len:
                sequences.append(seq)
    else:
        with open(path, "r") as f:
            for line in f:
                seq = clean_dna(line.strip())
                if len(seq) >= min_seq_len:
                    sequences.append(seq)

    if len(sequences) < 2:
        raise RuntimeError("Need at least 2 usable sequences after cleaning.")

    return sequences


def split_sequences(sequences, train_fraction=0.9):
    sequences = sequences[:]
    random.shuffle(sequences)

    split_idx = max(1, int(len(sequences) * train_fraction))
    split_idx = min(split_idx, len(sequences) - 1)

    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]
    return train_seqs, val_seqs


def sample_subsequence(seq: str, seq_len: int) -> str:
    if len(seq) < seq_len:
        raise ValueError(f"Sequence length {len(seq)} < requested seq_len {seq_len}")
    if len(seq) == seq_len:
        return seq
    start = random.randint(0, len(seq) - seq_len)
    return seq[start:start + seq_len]


def build_batch_from_sequences(tokenizer, seq_pool, batch_size: int, seq_len: int, device: str):
    tokenized = []
    for _ in range(batch_size):
        parent_seq = random.choice(seq_pool)
        sub_seq = sample_subsequence(parent_seq, seq_len)
        ids = tokenizer.tokenize(sub_seq)
        tokenized.append(ids)
    return torch.tensor(tokenized, dtype=torch.long, device=device)


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_targets = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_targets.view(-1)
    )


def next_token_accuracy(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    preds = logits[:, :-1, :].argmax(dim=-1)
    targets = input_ids[:, 1:]
    return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate(model, tokenizer, seq_pool, device, eval_batches, batch_size, seq_len, amp_dtype):
    model.eval()
    losses = []
    accs = []

    for _ in range(eval_batches):
        batch = build_batch_from_sequences(tokenizer, seq_pool, batch_size, seq_len, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = model(batch)
            loss = causal_lm_loss(logits, batch)

        losses.append(loss.detach().float().item())
        accs.append(next_token_accuracy(logits, batch))

    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accs) / len(accs)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity, avg_acc


def maybe_enable_gradient_checkpointing(model, use_gradient_checkpointing: bool):
    if not use_gradient_checkpointing:
        print("Gradient checkpointing disabled by config.")
        return

    enabled = False

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            enabled = True
            print("Enabled gradient checkpointing via model.gradient_checkpointing_enable().")
        except Exception as e:
            print(f"Could not enable gradient checkpointing that way: {e}")

    if (not enabled) and hasattr(model, "backbone"):
        try:
            model.backbone.gradient_checkpointing = True
            enabled = True
            print("Enabled gradient checkpointing via model.backbone.gradient_checkpointing = True.")
        except Exception as e:
            print(f"Could not set backbone gradient checkpointing directly: {e}")

    if not enabled:
        print("Warning: gradient checkpointing was not enabled.")


def maybe_load_locked_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        print("Initialization mode: UNLOCKED base Evo checkpoint")
        return

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"LOCKED_CKPT does not exist: {ckpt_path}")

    print("Initialization mode: LOCKED checkpoint")
    print(f"Loading locked checkpoint from: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Loaded locked checkpoint with strict=False")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("First 10 missing keys:")
        for k in missing[:10]:
            print(f"  {k}")

    if len(unexpected) > 0:
        print("First 10 unexpected keys:")
        for k in unexpected[:10]:
            print(f"  {k}")


def get_block_target_names(model, target_blocks):
    names = []
    for name, _ in model.named_parameters():
        parts = name.split(".")
        if len(parts) < 2 or parts[0] != "blocks":
            continue

        try:
            block_idx = int(parts[1])
        except ValueError:
            continue

        if block_idx in target_blocks:
            names.append(name)

    return names


def freeze_all_except_targets(model, target_names_set):
    for name, param in model.named_parameters():
        param.requires_grad = name in target_names_set


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def count_params_by_name(model, names):
    name_set = set(names)
    return sum(p.numel() for n, p in model.named_parameters() if n in name_set)


def fraction_str(numerator, denominator):
    if denominator == 0:
        return "0.00%"
    return f"{100.0 * numerator / denominator:.2f}%"


def estimate_tokens_seen(train_steps, batch_size, seq_len, grad_accum_steps):
    return train_steps * batch_size * seq_len * grad_accum_steps


def estimate_total_train_nt(train_seqs):
    return sum(len(x) for x in train_seqs)


def estimate_coverage_ratio(tokens_seen, total_train_nt):
    if total_train_nt == 0:
        return 0.0
    return tokens_seen / total_train_nt


def build_optimizer(model, optimizer_name: str, lr: float):
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name.lower() == "adamw":
        print("Using AdamW optimizer.")
        return optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

    if optimizer_name.lower() == "sgd":
        print("Using SGD optimizer.")
        return optim.SGD(trainable_params, lr=lr, momentum=0.9)

    raise ValueError(f"Unsupported OPTIMIZER_NAME: {optimizer_name}")


def load_evo_model(model_name: str, device: str):
    print("Loading Evo model...")
    evo_model = Evo(model_name)
    model = evo_model.model.to(device)
    tokenizer = evo_model.tokenizer
    return model, tokenizer


def save_history_csv(history, out_csv):
    if not history:
        return

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
