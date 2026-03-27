import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import math
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from evo import Evo


# =========================
# Config
# =========================

#RESULTS_DIR = "results/evo_lock_projections"
RESULTS_DIR = "results/evo_lock_projections_v7"
MODEL_NAME = "evo-1-8k-base"
DEVICE = "cuda:0"

DATA_PATH = "data/sequences.fasta"
SEED = 42
TRAIN_FRACTION = 0.9
MIN_SEQ_LEN = 128

#LOCK_STEPS = 30
#LOCK_LR = 1e-5
LOCK_STEPS = 50
LOCK_LR = 5e-5
#ALPHA = 0.8
ALPHA = 0.5
TOP_K = 1

BATCH_SIZE = 1
SEQ_LEN = 64

VAL_EVERY = 5
VAL_BATCHES = 4

SAVE_CHECKPOINT = True
MAX_GRAD_NORM = 1.0

#add for v5
TARGET_BLOCKS = {0, 1, 2, 3, 4, 5, 6, 7}
#

os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# Precision setup
# =========================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this script.")

BF16_OK = torch.cuda.is_bf16_supported()
if BF16_OK:
    AMP_DTYPE = torch.bfloat16
    USE_SCALER = False
else:
    AMP_DTYPE = torch.float16
    USE_SCALER = True


# =========================
# Data loading
# =========================
def clean_dna(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return "".join(c for c in seq if c in {"A", "C", "G", "T"})


def load_sequences(path: str):
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
                        if len(seq) >= MIN_SEQ_LEN:
                            sequences.append(seq)
                        current = []
                else:
                    current.append(line)

            if current:
                seq = clean_dna("".join(current))
                if len(seq) >= MIN_SEQ_LEN:
                    sequences.append(seq)
    else:
        with open(path, "r") as f:
            for line in f:
                seq = clean_dna(line.strip())
                if len(seq) >= MIN_SEQ_LEN:
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


# =========================
# Loss / metrics
# =========================
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


# =========================
# Target selection
# =========================

#def get_hyena_projection_targets(model):
#    targets = []
#    names = []
#
#    for name, param in model.named_parameters():
#        if name.endswith(".projections.weight"):
#            targets.append(param)
#            names.append(name)
#
#    return names, targets

#change for v5
def get_hyena_projection_targets(model):
    targets = []
    names = []

    for name, param in model.named_parameters():
        if not name.endswith(".projections.weight"):
            continue

        parts = name.split(".")
        if len(parts) < 3 or parts[0] != "blocks":
            continue

        block_idx = int(parts[1])
        if block_idx in TARGET_BLOCKS:
            targets.append(param)
            names.append(name)

    return names, targets

def freeze_non_targets(model, target_names_set):
    for name, param in model.named_parameters():
        param.requires_grad = name in target_names_set


# =========================
# Spectral term
# =========================
def spectral_term_topk_mean(target_params, top_k=1):
    vals = []

    for p in target_params:
        s = torch.linalg.svdvals(p.float())
        k = min(top_k, s.numel())
        vals.append(s[:k].mean())

    #return torch.stack(vals).mean()
    return torch.stack(vals).sum()


# =========================
# Eval
# =========================
@torch.no_grad()
def evaluate_retain(model, tokenizer, seq_pool, device, val_batches=4, batch_size=1, seq_len=64):
    model.eval()
    losses = []
    accs = []

    for _ in range(val_batches):
        batch = build_batch_from_sequences(tokenizer, seq_pool, batch_size, seq_len, device)

        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            logits, _ = model(batch)
            loss = causal_lm_loss(logits, batch)

        losses.append(loss.detach().float().item())
        accs.append(next_token_accuracy(logits, batch))

    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accs) / len(accs)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity, avg_acc


# =========================
# Locking
# =========================
def lock_evo_hyena_projections():
    print("Loading sequences...")
    sequences = load_sequences(DATA_PATH)
    train_seqs, val_seqs = split_sequences(sequences, TRAIN_FRACTION)

    print(f"Loaded {len(sequences)} usable sequences from {DATA_PATH}")
    print(f"Train sequences: {len(train_seqs)}")
    print(f"Val sequences:   {len(val_seqs)}")

    print("Loading Evo model...")
    evo_model = Evo(MODEL_NAME)
    model = evo_model.model.to(DEVICE)
    tokenizer = evo_model.tokenizer

    target_names, target_params = get_hyena_projection_targets(model)
    target_names_set = set(target_names)

    #add for v5
    print(f"Target block subset: {sorted(TARGET_BLOCKS)}")
    if len(target_params) == 0:
        raise RuntimeError("No target projection weights were selected.")
    #

    print(f"Found {len(target_params)} Hyena projection target matrices")
    for name in target_names:
        print(f"  {name}")

    freeze_non_targets(model, target_names_set)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Using AMP dtype: {AMP_DTYPE}")
    print(f"Using GradScaler: {USE_SCALER}")

    #optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LOCK_LR, momentum=0.9)
    #scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)

    #v6-AdamW optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LOCK_LR,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )


    history = []
    optimizer.zero_grad(set_to_none=True)

    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Starting locking for {LOCK_STEPS} steps...")
    print(f"ALPHA={ALPHA}, TOP_K={TOP_K}, LOCK_LR={LOCK_LR}")

    for step in tqdm(range(LOCK_STEPS)):
        model.train()

        batch = build_batch_from_sequences(tokenizer, train_seqs, BATCH_SIZE, SEQ_LEN, DEVICE)

        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            logits, _ = model(batch)
            retain_loss = causal_lm_loss(logits, batch)

        spec_term = spectral_term_topk_mean(target_params, top_k=TOP_K)

        lock_loss = ALPHA * retain_loss - (1.0 - ALPHA) * spec_term

        if USE_SCALER:
            scaler.scale(lock_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            lock_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if step % VAL_EVERY == 0 or step == LOCK_STEPS - 1:
            val_loss, val_ppl, val_acc = evaluate_retain(
                model,
                tokenizer,
                val_seqs,
                DEVICE,
                val_batches=VAL_BATCHES,
                batch_size=BATCH_SIZE,
                seq_len=SEQ_LEN,
            )

            record = {
                "step": step,
                "retain_loss": float(retain_loss.detach().float().item()),
                "spectral_term": float(spec_term.detach().float().item()),
                "lock_loss": float(lock_loss.detach().float().item()),
                "val_retain_loss": float(val_loss),
                "val_retain_ppl": float(val_ppl),
                "val_retain_acc": float(val_acc),
            }
            history.append(record)

            print(
                f"Step {step:03d} | "
                f"retain_loss={record['retain_loss']:.4f} | "
                f"spectral_term={record['spectral_term']:.4f} | "
                f"lock_loss={record['lock_loss']:.4f} | "
                f"val_retain_loss={record['val_retain_loss']:.4f} | "
                f"val_ppl={record['val_retain_ppl']:.4f} | "
                f"val_acc={record['val_retain_acc']:.4f}"
            )

    return model, history, total_params, trainable_params, target_names


# =========================
# Save
# =========================
def save_history_csv(history, out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "retain_loss",
                "spectral_term",
                "lock_loss",
                "val_retain_loss",
                "val_retain_ppl",
                "val_retain_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def save_lock_plot(history, out_png):
    steps = [x["step"] for x in history]
    retain_losses = [x["retain_loss"] for x in history]
    spectral_terms = [x["spectral_term"] for x in history]
    val_losses = [x["val_retain_loss"] for x in history]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, retain_losses, marker="o", label="Retain loss")
    plt.plot(steps, val_losses, marker="s", label="Val retain loss")
    plt.plot(steps, spectral_terms, marker="^", label="Spectral term")
    plt.xlabel("Lock step")
    plt.ylabel("Value")
    plt.title("Evo Hyena projection locking")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_summary(out_txt, total_params, trainable_params, target_names):
    with open(out_txt, "w") as f:
        f.write("Evo Hyena projection locking summary\n")
        f.write("==================================\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Data path: {DATA_PATH}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Lock steps: {LOCK_STEPS}\n")
        f.write(f"Lock LR: {LOCK_LR}\n")
        f.write(f"Alpha: {ALPHA}\n")
        f.write(f"Top-k: {TOP_K}\n")
        #add for v5
        f.write(f"Target blocks: {sorted(TARGET_BLOCKS)}\n")
        #
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Seq len: {SEQ_LEN}\n")
        f.write(f"AMP dtype: {AMP_DTYPE}\n")
        f.write(f"GradScaler enabled: {USE_SCALER}\n\n")
        f.write(f"Total params: {total_params:,}\n")
        f.write(f"Trainable params: {trainable_params:,}\n")
        f.write(f"Number of target matrices: {len(target_names)}\n\n")
        f.write("Target matrices:\n")
        for name in target_names:
            f.write(f"{name}\n")


# =========================
# Main
# =========================
if __name__ == "__main__":
    model, history, total_params, trainable_params, target_names = lock_evo_hyena_projections()

    csv_path = os.path.join(RESULTS_DIR, "lock_metrics.csv")
    png_path = os.path.join(RESULTS_DIR, "lock_curve.png")
    txt_path = os.path.join(RESULTS_DIR, "lock_summary.txt")
    ckpt_path = os.path.join(RESULTS_DIR, "model_locked_projections.pt")

    save_history_csv(history, csv_path)
    save_lock_plot(history, png_path)
    save_summary(txt_path, total_params, trainable_params, target_names)

    if SAVE_CHECKPOINT:
        torch.save(model.state_dict(), ckpt_path)

    print(f"\nSaved lock metrics to: {csv_path}")
    print(f"Saved lock plot to:    {png_path}")
    print(f"Saved lock summary to: {txt_path}")
    if SAVE_CHECKPOINT:
        print(f"Saved locked ckpt to:  {ckpt_path}")
