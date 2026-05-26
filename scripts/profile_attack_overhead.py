#!/usr/bin/env python
"""Profile GPU memory and per-step time overhead for each attack mechanism.

Loads the α=3×10⁴ locked model once per attack configuration, instruments
a single training step, and records:
  - Trainable parameter count
  - Extra parameters vs pretrained (absolute and %)
  - Peak GPU memory during forward+backward (MiB)
  - Extra GPU memory vs no-attack forward (MiB)
  - Mean step time over N_WARMUP+N_MEASURE steps (ms/step)

Usage:
  python scripts/profile_attack_overhead.py --device cuda:0 \
      --out results/attack_overhead.csv
"""
from __future__ import annotations
import os, sys, time, csv, argparse, math
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import (load_evo_model, maybe_load_locked_checkpoint,
                       load_sequences, get_amp_settings,
                       inject_bypass_layers, inject_theorem8_layers,
                       get_block_params, freeze_all_except)

N_WARMUP  = 3    # warmup steps (excluded from timing)
N_MEASURE = 10   # measured steps
SEQ_LEN   = 1024 # tokens per batch row
BATCH     = 1    # batch size (matches finetune.py default)
LOCKED_CKPT = "results/lock_alpha30k/model_specdef.pt"

# -----------------------------------------------------------------
# attack configurations to profile
# -----------------------------------------------------------------
ATTACKS = [
    # (label,  extra_kwargs to pass to load_model)
    ("Pretrained (no lock)",  dict(locked_ckpt=None)),
    ("Locked – no FT",        dict(locked_ckpt=LOCKED_CKPT, freeze_all=True)),
    ("Locked naive FT",       dict(locked_ckpt=LOCKED_CKPT)),
    ("LoRA (r=16)",           dict(locked_ckpt=LOCKED_CKPT, use_lora=True,
                                   lora_rank=16, lora_alpha=32, freeze_comp=True)),
    ("B-injection bypass",    dict(locked_ckpt=LOCKED_CKPT, layer_injection=True,
                                   freeze_comp=True)),
    ("SVD-chain k=1",         dict(locked_ckpt=LOCKED_CKPT, theorem8_injection=True,
                                   theorem8_k=1, freeze_comp=True)),
    ("SVD-chain k=3",         dict(locked_ckpt=LOCKED_CKPT, theorem8_injection=True,
                                   theorem8_k=3, freeze_comp=True)),
    ("SVD-chain k=5",         dict(locked_ckpt=LOCKED_CKPT, theorem8_injection=True,
                                   theorem8_k=5, freeze_comp=True)),
]


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_batch(tokenizer, device):
    """Random batch of BATCH×SEQ_LEN tokens."""
    ids = torch.randint(0, tokenizer.vocab_size, (BATCH, SEQ_LEN),
                        device=device, dtype=torch.long)
    return ids


def profile_one(label, load_kwargs, device, pretrained_trainable, base_mem_mib):
    """Returns a dict of metrics for one attack configuration."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # ---- Load model -------------------------------------------------------
    # We abuse the finetune setup: load base model then apply attack wiring.
    # This mirrors what finetune.py does.
    from src.utils import load_evo_model, maybe_load_locked_checkpoint
    import types, copy

    locked_ckpt = load_kwargs.get("locked_ckpt")
    freeze_all  = load_kwargs.get("freeze_all", False)
    use_lora    = load_kwargs.get("use_lora", False)
    layer_inj   = load_kwargs.get("layer_injection", False)
    thm8        = load_kwargs.get("theorem8_injection", False)
    thm8_k      = load_kwargs.get("theorem8_k", 3)
    freeze_comp = load_kwargs.get("freeze_comp", False)
    lora_rank   = load_kwargs.get("lora_rank", 16)
    lora_alpha  = load_kwargs.get("lora_alpha", 32)

    MODEL_NAME = "evo-1-8k-base"
    model, tokenizer = load_evo_model(MODEL_NAME, device)

    # 1. Load locked checkpoint (2-arg API)
    if locked_ckpt:
        maybe_load_locked_checkpoint(model, locked_ckpt)

    # 2. Inject attacks (mirrors finetune.py _run_finetune order)
    if layer_inj:
        inject_bypass_layers(model)

    if thm8:
        inject_theorem8_layers(model, k=thm8_k)

    # 3. Determine trainable parameter set
    if freeze_all:
        for p in model.parameters():
            p.requires_grad_(False)
        total, trainable = count_params(model)
    else:
        # Mirror finetune.py: get block params, apply freeze_comp, then freeze_all_except
        target_names_set = set(get_block_params(model, list(range(32))))

        _COMP_SUFFIXES = (".comp.weight", ".specdef.comp.weight")
        _LINEAR_SUFFIXES = (".specdef.linear.weight", ".fused.weight")

        if layer_inj:
            target_names_set = {n for n in target_names_set
                                if not any(n.endswith(s) for s in _LINEAR_SUFFIXES)}
            bypass_names = {n for n, _ in model.named_parameters()
                            if n.endswith(".bypass.weight")}
            target_names_set |= bypass_names

        if freeze_comp:
            target_names_set = {n for n in target_names_set
                                 if not any(n.endswith(s) for s in _COMP_SUFFIXES)}
            # Cast C to bf16 to save memory (frozen, norm. frozen, matches finetune.py)
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
        total, trainable = count_params(model)
    extra_params = trainable - pretrained_trainable

    # ---- Measure memory for one forward pass (no grad) --------------------
    torch.cuda.reset_peak_memory_stats(device)
    batch = make_batch(tokenizer, device)
    amp_dtype = torch.bfloat16
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            _out, _ = model(batch)
    fwd_mem = torch.cuda.max_memory_allocated(device) / 1024**2

    # ---- Measure memory + time for forward+backward -----------------------
    # Use bitsandbytes 8-bit AdamW to match finetune.py exactly
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-6) if trainable > 0 else None
    except ImportError:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-6) if trainable > 0 else None

    step_times = []
    torch.cuda.reset_peak_memory_stats(device)

    for i in range(N_WARMUP + N_MEASURE):
        batch = make_batch(tokenizer, device)
        t0 = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = model(batch)
            # causal LM loss on the batch
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
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        if i >= N_WARMUP:
            step_times.append(elapsed)

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    mean_step_ms = sum(step_times) / len(step_times)

    del model, batch
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    return {
        "label":             label,
        "total_params_M":    round(total / 1e6, 1),
        "trainable_params_M": round(trainable / 1e6, 1),
        "extra_params_M":    round(extra_params / 1e6, 1),
        "extra_params_pct":  round(100 * extra_params / max(total, 1), 2),
        "fwd_mem_MiB":       round(fwd_mem, 0),
        "fwd_mem_extra_MiB": round(fwd_mem - base_mem_mib, 0),
        "peak_mem_MiB":      round(peak_mem, 0),
        "step_ms":           round(mean_step_ms, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/attack_overhead.csv")
    args = ap.parse_args()
    device = args.device

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ---- Pretrained baseline (needed for relative metrics) ----------------
    print("=== Profiling pretrained baseline ===")
    from src.utils import load_evo_model
    model0, tok0 = load_evo_model("evo-1-8k-base", device)
    pretrained_trainable = sum(p.numel() for p in model0.parameters() if p.requires_grad)

    torch.cuda.reset_peak_memory_stats(device)
    batch0 = torch.randint(0, tok0.vocab_size, (BATCH, SEQ_LEN), device=device)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _ = model0(batch0)
    base_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"  pretrained_trainable={pretrained_trainable:,}  base_fwd_mem={base_mem:.0f} MiB")
    del model0
    torch.cuda.empty_cache()

    # ---- Profile all attacks (each in isolated subprocess) ----------------
    rows = []
    for label, kwargs in ATTACKS:
        print(f"\n=== {label} ===")
        # Write kwargs to a temp file and launch a subprocess to avoid
        # CUDA memory fragmentation between configs
        import subprocess, tempfile, pickle
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            tmp_in = tf.name
            pickle.dump({"label": label, "kwargs": kwargs,
                         "device": device,
                         "pretrained_trainable": pretrained_trainable,
                         "base_mem_mib": base_mem}, tf)
        tmp_out = tmp_in + ".out.pkl"
        worker_path = os.path.join(ROOT, "scripts", "profile_attack_overhead_worker.py")
        script = (
            "import sys, pickle, os; "
            "sys.path.insert(0, %r); "
            "sys.path.insert(0, %r); "
            "os.chdir(%r); "
            "exec(open(%r).read()); "
            "run_worker(%r, %r)"
        ) % (ROOT, os.path.join(ROOT, "scripts"), ROOT, worker_path, tmp_in, tmp_out)
        env = os.environ.copy()
        env["PYTHONPATH"] = ROOT
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env,
            cwd=ROOT
        )
        if os.path.exists(tmp_out):
            with open(tmp_out, "rb") as f:
                row = pickle.load(f)
            os.unlink(tmp_out)
        else:
            row = {"label": label, "error": result.stderr[-500:] if result.stderr else "no output"}
        os.unlink(tmp_in)
        rows.append(row)
        if "error" in row:
            print(f"  ERROR: {row['error'][-200:]}")
        else:
            print(f"  trainable={row['trainable_params_M']}M (+{row['extra_params_M']}M)"
                  f"  peak_mem={row['peak_mem_MiB']:.0f}MiB"
                  f"  step={row['step_ms']}ms")

    # ---- Write CSV --------------------------------------------------------
    if rows:
        fields = [k for k in rows[0] if k != "error"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.out}")

    # ---- Print table -------------------------------------------------------
    print("\n=== Attack overhead summary ===")
    print(f"{'Mechanism':30s}  {'Train.M':>8s}  {'Extra%':>7s}  {'PeakMiB':>8s}  {'ms/step':>8s}")
    for r in rows:
        if "error" in r:
            print(f"  {r['label']:30s}  ERROR")
            continue
        print(f"  {r['label']:30s}  {r['trainable_params_M']:8.1f}  "
              f"{r['extra_params_pct']:7.2f}  {r['peak_mem_MiB']:8.0f}  {r['step_ms']:8.1f}")


if __name__ == "__main__":
    main()
