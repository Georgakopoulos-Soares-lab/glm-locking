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

    target_blocks: set = field(default_factory=lambda: set(range(8)))
    target_layer_patterns: tuple = (
        ".projections.weight",
        ".out_filter_dense.weight",
        ".mlp.l1.weight",
        ".mlp.l2.weight",
        ".mlp.l3.weight",
    )

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
    """Return a list of dicts with SVD stats for each target matrix.

    Uses randomized SVD (svd_lowrank) on CPU in a ThreadPoolExecutor so all
    matrices are processed in parallel.  Full SVD of 4096×4096 matrices on
    CPU is O(n³) and would take hours; svd_lowrank is O(q·n) and only needs
    the top-k values.  Condition number is not computed (set to -1) because
    it requires the smallest singular value which svd_lowrank does not provide.
    """
    from concurrent.futures import ThreadPoolExecutor

    q = max(top_k, 3)

    # Pre-copy all tensors to CPU float32 on the main thread to avoid
    # concurrent GPU→CPU PCIe transfers thrashing the bus
    cpu_mats = [(name, p.detach().float().cpu(), list(p.shape)) for name, p in zip(names, params)]

    def _one(args):
        name, mat, shape = args
        if min(mat.shape) <= q:
            sv = torch.linalg.svdvals(mat)
        else:
            _, sv, _ = torch.svd_lowrank(mat, q=q, niter=2)
        k = min(top_k, sv.numel())
        return {
            "name": name,
            "shape": shape,
            "sigma_1": float(sv[0]),
            "sigma_2": float(sv[1]) if sv.numel() > 1 else 0.0,
            "sigma_3": float(sv[2]) if sv.numel() > 2 else 0.0,
            "top_k_mean": float(sv[:k].mean()),
            "condition_number": -1.0,  # not computed — svd_lowrank only gives top-k values
        }

    with ThreadPoolExecutor(max_workers=min(len(cpu_mats), 16)) as pool:
        results = list(pool.map(_one, cpu_mats))
    return results


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

@torch.no_grad()
def spectral_term_topk(target_params: list[torch.Tensor], top_k: int = 1) -> torch.Tensor:
    """Mean of top-k singular values across all target matrices (GPU sequential, logging only).

    Processes matrices one by one on GPU using svd_lowrank.  Peak scratch per matrix is
    O(n*q) which is small (q=2 columns) and fits in the ~1 GiB free on a loaded A100.
    Much faster than the CPU+ThreadPoolExecutor approach because no PCIe data movement.
    """
    q = max(top_k + 1, 2)
    vals = []
    for p in target_params:
        mat = p.detach().float()          # fp32 on GPU, no CPU copy
        _, s, _ = torch.svd_lowrank(mat, q=q, niter=2)
        vals.append(s[:top_k].mean())
    return torch.stack(vals).mean()


@torch.no_grad()
def add_spectral_grad_inplace(
    target_params: list[torch.Tensor],
    top_k: int,
    spec_loss_coeff: float,
) -> float:
    """Compute the analytical spectral gradient and add it directly to param.grad.

    Returns mean top-k singular value across all target matrices (for logging).

    Instead of building an autograd graph through 154 CPU SVDs, this function computes
    the gradient analytically:
        d(mean_topk_SV(A)) / d(A) = (1/k) * U[:, :k] @ V[:, :k].T
    where spec_loss = spec_loss_coeff * (1/N) * sum_i mean_topk_SV(A_i).

    Benefits over the old autograd approach:
    - No GPU→CPU PCIe transfers (29 GB/step avoided)
    - No ThreadPoolExecutor CPU contention
    - No autograd graph overhead for 154-node SVD backward
    - Fully GPU-resident; peak scratch ≈ 200 MB (one fp32 matrix copy + O(n*q) scratch)

    DDP note: DDP all_reduces these grads together with retain grads during
    retain_loss.backward().  Since spec grads are identical on all ranks, the
    all_reduce (averaging) preserves them unchanged.  Do NOT divide by world_size.
    """
    q = max(top_k, 2)
    trainable = [p for p in target_params if p.requires_grad]
    if not trainable:
        return 0.0

    N = len(trainable)
    spec_vals = []

    for p in trainable:
        mat = p.detach().float()                  # fp32 GPU copy
        U, s, V = torch.svd_lowrank(mat, q=q, niter=2)   # U:[n,q], V:[m,q], s:[q]
        k = min(top_k, s.numel())
        spec_vals.append(float(s[:k].mean()))

        # Analytical gradient: (spec_loss_coeff / N / k) * U[:, :k] @ V[:, :k].T
        g = (spec_loss_coeff / (N * k)) * (U[:, :k] @ V[:, :k].t())
        g = g.to(dtype=p.dtype)                   # match param dtype (bf16 or fp32)

        if p.grad is None:
            p.grad = g
        else:
            p.grad.add_(g)

    return float(sum(spec_vals) / len(spec_vals))


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

    # Try the standard HuggingFace API first
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled (gradient_checkpointing_enable).")
            return
        except Exception:
            pass

    # StripedHyena (Evo): monkey-patch stateless_forward to checkpoint each block.
    # The model's stateless_forward iterates over self.blocks in a plain loop;
    # we replace it with a version that wraps each block call in
    # torch.utils.checkpoint.checkpoint(), which discards intermediate activations
    # and recomputes them during backward — halving activation memory.
    import types
    from torch.utils.checkpoint import checkpoint as ckpt_fn

    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "stateless_forward") and hasattr(backbone, "blocks"):
        original_stateless = backbone.stateless_forward

        def _checkpointed_stateless(self, x, padding_mask=None):
            if type(padding_mask) == torch.Tensor:
                x = x * padding_mask[..., None]
            for block in self.blocks:
                # checkpoint requires all inputs to be tensors; pass padding_mask
                # as a dummy tensor when None so the signature is consistent.
                def _block_fn(x, _block=block):
                    out, _ = _block(x, inference_params=None, padding_mask=None)
                    return out
                x = ckpt_fn(_block_fn, x, use_reentrant=False)
            return x, None

        backbone.stateless_forward = types.MethodType(_checkpointed_stateless, backbone)
        print("Gradient checkpointing enabled (StripedHyena stateless_forward patched).")
        return

    print("Warning: gradient checkpointing not supported by this model — using batch_size=1 is recommended to avoid OOM.")


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model, name: str, lr: float):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if name.lower() == "adamw":
        return optim.AdamW(trainable, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
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
