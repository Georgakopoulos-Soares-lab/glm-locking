"""Worker script invoked by profile_attack_overhead.py in an isolated subprocess.

Each invocation loads the model, runs one attack configuration, writes the
result to a pickle file, and exits — completely releasing all CUDA memory.
"""
from __future__ import annotations
import os, sys, time, pickle
import torch
import torch.nn.functional as F


N_WARMUP  = 3
N_MEASURE = 10
SEQ_LEN   = 1024
BATCH     = 1


def run_worker(tmp_in: str, tmp_out: str) -> None:
    with open(tmp_in, "rb") as f:
        d = pickle.load(f)

    label               = d["label"]
    load_kwargs         = d["kwargs"]
    device              = d["device"]
    pretrained_trainable = d["pretrained_trainable"]
    base_mem_mib        = d["base_mem_mib"]

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    result = _profile_one(label, load_kwargs, device, pretrained_trainable, base_mem_mib)
    with open(tmp_out, "wb") as f:
        pickle.dump(result, f)


def _count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _make_batch(tokenizer, device):
    return torch.randint(0, tokenizer.vocab_size, (BATCH, SEQ_LEN),
                         device=device, dtype=torch.long)


def _profile_one(label, load_kwargs, device, pretrained_trainable, base_mem_mib):
    from src.utils import (load_evo_model, maybe_load_locked_checkpoint,
                           inject_bypass_layers, inject_theorem8_layers,
                           get_block_params, freeze_all_except)

    # initialize CUDA in this subprocess before any CUDA memory API calls
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    locked_ckpt = load_kwargs.get("locked_ckpt")
    freeze_all  = load_kwargs.get("freeze_all", False)
    use_lora    = load_kwargs.get("use_lora", False)
    layer_inj   = load_kwargs.get("layer_injection", False)
    thm8        = load_kwargs.get("theorem8_injection", False)
    thm8_k      = load_kwargs.get("theorem8_k", 3)
    freeze_comp = load_kwargs.get("freeze_comp", False)
    lora_rank   = load_kwargs.get("lora_rank", 16)
    lora_alpha  = load_kwargs.get("lora_alpha", 32)

    model, tokenizer = load_evo_model("evo-1-8k-base", device)

    # Enable gradient checkpointing to match finetune.py (default: True)
    from src.utils import maybe_enable_gradient_checkpointing
    maybe_enable_gradient_checkpointing(model, True)

    if locked_ckpt:
        maybe_load_locked_checkpoint(model, locked_ckpt)

    if layer_inj:
        inject_bypass_layers(model)

    if thm8:
        inject_theorem8_layers(model, k=thm8_k)

    _COMP_SUFFIXES   = (".comp.weight", ".specdef.comp.weight")
    _LINEAR_SUFFIXES = (".specdef.linear.weight", ".fused.weight")

    if freeze_all:
        for p in model.parameters():
            p.requires_grad_(False)
        total, trainable = _count_params(model)
    else:
        target_names_set = set(get_block_params(model, list(range(32))))

        if layer_inj:
            target_names_set = {n for n in target_names_set
                                if not any(n.endswith(s) for s in _LINEAR_SUFFIXES)}
            bypass_names = {n for n, _ in model.named_parameters()
                            if n.endswith(".bypass.weight")}
            target_names_set |= bypass_names

        if freeze_comp:
            target_names_set = {n for n in target_names_set
                                 if not any(n.endswith(s) for s in _COMP_SUFFIXES)}
            for name, param in model.named_parameters():
                if any(name.endswith(s) for s in _COMP_SUFFIXES):
                    if param.dtype != torch.float64:
                        param.data = param.data.to(torch.bfloat16)

        if use_lora:
            from src.lora import inject_lora
            _lora_targets = ("projections", "Wqkv", "out_proj", "mlp.l")
            _, lora_names = inject_lora(model, rank=lora_rank, alpha=lora_alpha,
                                        target_substrings=_lora_targets)
            target_names_set = set(lora_names)

        freeze_all_except(model, target_names_set)
        total, trainable = _count_params(model)

    extra_params = trainable - pretrained_trainable

    # Forward-only memory
    amp_dtype = torch.bfloat16
    torch.cuda.reset_peak_memory_stats(device)
    batch = _make_batch(tokenizer, device)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            _out, _ = model(batch)
    fwd_mem = torch.cuda.max_memory_allocated(device) / 1024**2

    # Optimizer — use 8-bit AdamW to match finetune.py
    if trainable > 0:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                [p for p in model.parameters() if p.requires_grad], lr=1e-6)
        except ImportError:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-6)
    else:
        optimizer = None

    step_times = []
    torch.cuda.reset_peak_memory_stats(device)

    for i in range(N_WARMUP + N_MEASURE):
        batch = _make_batch(tokenizer, device)
        t0 = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = model(batch)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        if i >= N_WARMUP:
            step_times.append((time.perf_counter() - t0) * 1000)

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    mean_step_ms = sum(step_times) / len(step_times)

    return {
        "label":              label,
        "total_params_M":     round(total / 1e6, 1),
        "trainable_params_M": round(trainable / 1e6, 1),
        "extra_params_M":     round(extra_params / 1e6, 1),
        "extra_params_pct":   round(100 * extra_params / max(total, 1), 2),
        "fwd_mem_MiB":        round(fwd_mem, 0),
        "fwd_mem_extra_MiB":  round(fwd_mem - base_mem_mib, 0),
        "peak_mem_MiB":       round(peak_mem, 0),
        "step_ms":            round(mean_step_ms, 1),
    }
