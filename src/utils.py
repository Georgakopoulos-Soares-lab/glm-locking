"""Shared utilities for evo-locking experiments."""

import os
import csv
import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LockConfig:
    """Configuration for the SpecDef locking procedure."""
    run_name: str
    results_dir: str
    model_name: str = "evo-1-8k-base"
    device: str = "cuda:0"
    retain_data_path: str = "data/retain.fasta"
    seed: int = 42
    train_fraction: float = 0.9
    min_seq_len: int = 512

    lock_steps: int = 500
    lock_lr: float = 5e-5
    alpha_start: float = 0.8
    alpha_end: float = 0.3
    top_k: int = 1
    batch_size: int = 1
    seq_len: int = 512
    grad_accum_steps: int = 4
    val_every: int = 25
    val_batches: int = 8
    max_grad_norm: float = 1.0
    optimizer_name: str = "adafactor"

    target_blocks: set = field(default_factory=lambda: set(range(8)))
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


@dataclass
class FinetuneConfig:
    """Configuration for the fine-tuning (attack) procedure."""
    run_name: str
    results_dir: str
    model_name: str = "evo-1-8k-base"
    device: str = "cuda:0"
    data_path: str = "data/attack.fasta"
    seed: int = 42
    train_fraction: float = 0.9
    min_seq_len: int = 512

    train_steps: int = 5000
    lr: float = 1e-5
    batch_size: int = 1
    seq_len: int = 1024
    grad_accum_steps: int = 4
    val_every: int = 100
    eval_batches: int = 16
    max_grad_norm: float = 1.0
    optimizer_name: str = "adamw"

    target_blocks: set = field(default_factory=lambda: set(range(8)))
    locked_ckpt: str | None = None

    save_checkpoint: bool = True
    use_gradient_checkpointing: bool = True


@dataclass
class EvalConfig:
    """Configuration for pretrained model evaluation."""
    run_name: str
    results_dir: str
    model_name: str = "evo-1-8k-base"
    device: str = "cuda:0"
    data_path: str = "data/attack.fasta"
    seed: int = 42
    train_fraction: float = 0.9
    min_seq_len: int = 512
    seq_len: int = 1024
    eval_batches: int = 16
    batch_size: int = 1
    use_gradient_checkpointing: bool = True


# ---------------------------------------------------------------------------
# Seed / AMP
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_settings():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


# ---------------------------------------------------------------------------
# DDP (Distributed Data Parallel)
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int]:
    """Initialize DDP if launched via torchrun.

    Returns (rank, local_rank, world_size).
    Single-GPU fallback returns (0, 0, 1).
    """
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def cleanup_ddp():
    """Destroy the DDP process group if it was initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def wrap_ddp(model, local_rank: int, skip_ddp: bool = False):
    """Wrap model in DistributedDataParallel if distributed.

    If skip_ddp=True, skips wrapping but still returns (model, raw_model)
    so callers can use manual gradient sync instead (saves ~12 GB gradient buffers).

    Returns (wrapped_model, raw_model).  *raw_model* is always the
    unwrapped module so callers can access .state_dict() etc.
    """
    raw = model
    if torch.distributed.is_initialized() and not skip_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
        )
    return model, raw


def manual_allreduce_grads(model):
    """Average gradients across all ranks — lightweight alternative to DDP.

    Only touches parameters with .grad set, avoids DDP's pre-allocated
    gradient buckets (~12 GB for 6.4B params).
    """
    if not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    for p in model.parameters():
        if p.grad is not None:
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad.div_(world_size)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def clean_dna(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return "".join(c for c in seq if c in {"A", "C", "G", "T"})


def load_sequences(path: str, min_seq_len: int) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path does not exist: {path}")

    sequences: list[str] = []

    if path.endswith((".fa", ".fasta", ".fna")):
        current: list[str] = []
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


def split_sequences(sequences: list[str], train_fraction: float = 0.9):
    sequences = sequences[:]
    random.shuffle(sequences)
    split_idx = max(1, int(len(sequences) * train_fraction))
    split_idx = min(split_idx, len(sequences) - 1)
    return sequences[:split_idx], sequences[split_idx:]


def sample_subsequence(seq: str, seq_len: int) -> str:
    if len(seq) <= seq_len:
        return seq
    start = random.randint(0, len(seq) - seq_len)
    return seq[start : start + seq_len]


def build_batch(tokenizer, seq_pool, batch_size: int, seq_len: int, device: str):
    tokenized = []
    for _ in range(batch_size):
        parent_seq = random.choice(seq_pool)
        sub_seq = sample_subsequence(parent_seq, seq_len)
        ids = tokenizer.tokenize(sub_seq)
        tokenized.append(ids)
    return torch.tensor(tokenized, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------

def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_targets = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_targets.view(-1),
    )


def next_token_accuracy(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    preds = logits[:, :-1, :].argmax(dim=-1)
    targets = input_ids[:, 1:]
    return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate(model, tokenizer, seq_pool, device, eval_batches, batch_size, seq_len, amp_dtype):
    model.eval()
    losses, accs = [], []
    for _ in range(eval_batches):
        batch = build_batch(tokenizer, seq_pool, batch_size, seq_len, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = model(batch)
            loss = causal_lm_loss(logits, batch)
        losses.append(loss.detach().float().item())
        accs.append(next_token_accuracy(logits, batch))

    avg_loss = sum(losses) / len(losses)
    return avg_loss, math.exp(avg_loss), sum(accs) / len(accs)


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def get_lock_targets(model, target_blocks: set, layer_patterns: tuple):
    """Select all nn.Linear weights in the target blocks matching any pattern."""
    names, params = [], []
    for name, param in model.named_parameters():
        parts = name.split(".")
        if len(parts) < 3 or parts[0] != "blocks":
            continue
        try:
            block_idx = int(parts[1])
        except ValueError:
            continue
        if block_idx not in target_blocks:
            continue
        if any(name.endswith(pat) for pat in layer_patterns):
            names.append(name)
            params.append(param)
    return names, params


def get_block_params(model, target_blocks: set):
    """Select ALL parameters in the target blocks (for fine-tuning attack)."""
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


def freeze_all_except(model, target_names_set: set):
    for name, param in model.named_parameters():
        param.requires_grad = name in target_names_set


# ---------------------------------------------------------------------------
# SVD monitoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_svd_stats(names: list[str], params: list[torch.Tensor], top_k: int = 3):
    """Return a list of dicts with SVD stats for each target matrix."""
    stats = []
    for name, p in zip(names, params):
        sv = torch.linalg.svdvals(p.float())
        k = min(top_k, sv.numel())
        stats.append({
            "name": name,
            "shape": list(p.shape),
            "sigma_1": float(sv[0]),
            "sigma_2": float(sv[1]) if sv.numel() > 1 else 0.0,
            "sigma_3": float(sv[2]) if sv.numel() > 2 else 0.0,
            "top_k_mean": float(sv[:k].mean()),
            "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        })
    return stats


def log_svd_stats(stats: list[dict], label: str, filepath: str):
    """Write SVD stats to a CSV file."""
    rows = []
    for s in stats:
        rows.append({
            "phase": label,
            "name": s["name"],
            "shape": str(s["shape"]),
            "sigma_1": s["sigma_1"],
            "sigma_2": s["sigma_2"],
            "sigma_3": s["sigma_3"],
            "top_k_mean": s["top_k_mean"],
            "condition_number": s["condition_number"],
        })

    write_header = not os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Spectral term
# ---------------------------------------------------------------------------

def spectral_term_topk(target_params: list[torch.Tensor], top_k: int = 1) -> torch.Tensor:
    """Mean of top-k singular values across all target matrices.

    For logging only — call under torch.no_grad().
    Uses randomized SVD (svd_lowrank) — O(m*n*k) vs full SVD O(m*n*min(m,n)).
    """
    vals = []
    q = max(2 * top_k + 10, 20)
    for p in target_params:
        _, S, _ = torch.svd_lowrank(p.detach().float(), q=q, niter=2)
        vals.append(S[:top_k].mean())
    return torch.stack(vals).mean()


def add_spectral_grads(
    target_params: list,
    top_k: int,
    coeff: float,
    grad_scale: float = 1.0,
) -> float:
    """Add spectral loss gradients directly to p.grad — no autograd graph.

    Uses the analytical gradient: d(mean_i sigma_i(W)) / dW = (1/k) * sum_i u_i @ v_i^T
    Runs entirely under torch.no_grad(), allocating only one matrix at a time.

    The contribution to p.grad is: -coeff * grad_scale * (1/k) * sum_i u_i @ v_i^T
    Negated because the optimizer minimises (we want to maximise singular values).
    grad_scale should equal scaler.get_scale() when using GradScaler so that
    scaler.unscale_() correctly unscales SVD and retain gradients together.

    Returns:
        float — mean spectral term value across all target matrices (for logging).
    """
    q = max(2 * top_k + 10, 20)
    sv_sum = 0.0
    with torch.no_grad():
        for p in target_params:
            U, S, V = torch.svd_lowrank(p.detach().float(), q=q, niter=2)
            sv_sum += S[:top_k].mean().item()
            # d(-spec_term)/dW = -(1/k) sum_i u_i @ v_i^T
            sv_grad = (U[:, :top_k] @ V[:, :top_k].T) / top_k  # (m, n) float32
            contribution = (-coeff * grad_scale * sv_grad).to(p.dtype)
            if p.grad is None:
                p.grad = contribution.clone()
            else:
                p.grad.add_(contribution)
    return sv_sum / len(target_params)


# ---------------------------------------------------------------------------
# Alpha scheduling
# ---------------------------------------------------------------------------

def get_alpha(step: int, total_steps: int, alpha_start: float, alpha_end: float) -> float:
    """Linear schedule from alpha_start to alpha_end."""
    if total_steps <= 1:
        return alpha_start
    t = step / (total_steps - 1)
    return alpha_start + (alpha_end - alpha_start) * t


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_evo_model(model_name: str, device: str):
    from evo import Evo
    print(f"Loading Evo model '{model_name}'...")
    evo_model = Evo(model_name)
    model = evo_model.model.to(device)
    return model, evo_model.tokenizer


def maybe_load_locked_checkpoint(model, ckpt_path: str | None):
    if ckpt_path is None:
        print("Initialization: pretrained (unlocked)")
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Locked checkpoint not found: {ckpt_path}")
    print(f"Loading locked checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


def maybe_enable_gradient_checkpointing(model, enable: bool):
    if not enable:
        return
    # Try HuggingFace API first
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled.")
            return
        except Exception:
            pass
    # Manual: monkey-patch each block's forward with torch.utils.checkpoint
    if hasattr(model, "blocks"):
        from torch.utils.checkpoint import checkpoint as _ckpt

        def _make_ckpt_forward(orig_fwd):
            def wrapper(*args, **kwargs):
                return _ckpt(orig_fwd, *args, use_reentrant=False, **kwargs)
            return wrapper

        count = 0
        for block in model.blocks:
            block.forward = _make_ckpt_forward(block.forward)
            count += 1
        print(f"Gradient checkpointing enabled (manual, {count} blocks).")
        return
    print("Warning: gradient checkpointing was not enabled.")


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model, name: str, lr: float):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if name.lower() == "adamw":
        return optim.AdamW(trainable, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    if name.lower() == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(
            trainable, lr=lr,
            scale_parameter=False,
            relative_step=False,
            weight_decay=0.0,
        )
    if name.lower() == "sgd":
        return optim.SGD(trainable, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_history_csv(history: list[dict], path: str):
    if not history:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_params_by_name(model, names: list[str]):
    name_set = set(names)
    return sum(p.numel() for n, p in model.named_parameters() if n in name_set)
