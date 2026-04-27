"""Fine-tune (attack) Evo blocks — runs with locked or unlocked initialization.

This is the "attacker" step: tries to repurpose the model for a new task.
Compare locked vs unlocked checkpoints to measure lock effectiveness.

Usage:
    # Via YAML config (preferred):
    python scripts/finetune.py --config configs/ft_v10.yaml               # mode from YAML
    python scripts/finetune.py --config configs/ft_v10.yaml --locked      # override: locked only
    python scripts/finetune.py --config configs/ft_v10.yaml --unlocked    # override: unlocked only

    # Via hardcoded defaults:
    python scripts/finetune.py --locked
    python scripts/finetune.py --unlocked
"""

import os
import sys
import time
import argparse
from contextlib import nullcontext

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    FinetuneConfig,
    set_seed,
    get_amp_settings,
    setup_ddp,
    cleanup_ddp,
    wrap_ddp,
    load_sequences,
    split_sequences,
    build_batch,
    causal_lm_loss,
    next_token_accuracy,
    evaluate,
    specdef_fused_eval,
    get_block_params,
    get_lock_targets,
    freeze_all_except,
    load_evo_model,
    maybe_load_locked_checkpoint,
    maybe_enable_gradient_checkpointing,
    inject_bypass_layers,
    inject_theorem8_layers,
    build_optimizer,
    save_history_csv,
    count_params,
    count_trainable,
    count_params_by_name,
    compute_svd_stats,
)

# ===========================================================================
# Shared hyperparameters (identical for locked and unlocked — fair comparison)
# ===========================================================================
_SHARED = dict(
    data_path="data/attack.fasta",
    val_data_path=None,  # if set, use as val (held-out); train_fraction then ignored for val split
    model_name="evo-1-8k-base",
    device="cuda:0",
    seed=42,
    train_fraction=0.9,
    min_seq_len=1024,

    train_steps=5000,
    lr=1e-5,
    batch_size=1,
    seq_len=1024,
    grad_accum_steps=4,
    val_every=100,
    eval_batches=16,
    max_grad_norm=1.0,
    optimizer_name="adamw",

    # Attack ALL 32 blocks (must match what was locked)
    target_blocks=set(range(32)),

    save_checkpoint=True,
    use_gradient_checkpointing=True,
)

LOCKED_CONFIG = FinetuneConfig(
    run_name="ft_attack_locked_v10",
    results_dir="results/ft_attack_locked_v10",
    locked_ckpt="results/lock_v10/model_locked.pt",
    **_SHARED,
)

UNLOCKED_CONFIG = FinetuneConfig(
    run_name="ft_attack_unlocked_v10",
    results_dir="results/ft_attack_unlocked_v10",
    locked_ckpt=None,
    **_SHARED,
)


# ===========================================================================
# YAML config loader
# ===========================================================================
def _load_config(path: str, mode: str | None = None) -> list[FinetuneConfig]:
    """Load FinetuneConfig(s) from a YAML file.

    YAML must have a 'mode' field: 'locked', 'unlocked', or 'both'.
    The `mode` argument overrides the YAML field if provided.

    For mode='both', two configs are returned (locked first, unlocked second).
    The locked variant gets run_name + '_locked', the unlocked gets '_unlocked'.
    locked_ckpt is required in the YAML when mode includes locked.
    When mode includes unlocked, locked_ckpt is set to None for that config.
    """
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f)

    yaml_mode = d.pop("mode", "both")
    effective_mode = mode or yaml_mode

    # Auto-derive results_dir from run_name
    base_run_name = d["run_name"]

    if "target_blocks" in d:
        val = d["target_blocks"]
        d["target_blocks"] = set(range(val)) if isinstance(val, int) else set(val)
    # YAML loads scientific notation as string (e.g. '5e-5') — cast to float
    for float_key in ("lr", "weight_decay", "warmup_fraction", "theorem8_factor_lr"):
        if float_key in d:
            d[float_key] = float(d[float_key])

    locked_ckpt = d.pop("locked_ckpt", None)

    # Drop keys not in the dataclass
    from dataclasses import fields as _fields
    valid = {f.name for f in _fields(FinetuneConfig)}
    d = {k: v for k, v in d.items() if k in valid}

    configs = []
    if effective_mode in ("locked", "both"):
        if not locked_ckpt:
            raise ValueError(f"mode={effective_mode} requires 'locked_ckpt' in YAML")
        ld = {**d,
              "run_name": f"{base_run_name}_locked",
              "results_dir": f"results/{base_run_name}_locked",
              "locked_ckpt": locked_ckpt}
        configs.append(FinetuneConfig(**ld))
    if effective_mode in ("unlocked", "both"):
        ud = {**d,
              "run_name": f"{base_run_name}_unlocked",
              "results_dir": f"results/{base_run_name}_unlocked",
              "locked_ckpt": None}
        configs.append(FinetuneConfig(**ud))

    if not configs:
        raise ValueError(f"Invalid mode: {effective_mode}. Use 'locked', 'unlocked', or 'both'.")
    return configs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune (attack) Evo")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--locked", action="store_true", help="Attack locked checkpoint")
    group.add_argument("--unlocked", action="store_true", help="Attack pretrained (unlocked) baseline")
    args, _ = parser.parse_known_args()

    # --- DDP setup (once for the whole process) ---
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)

    if args.config:
        mode = "locked" if args.locked else ("unlocked" if args.unlocked else None)
        configs = _load_config(args.config, mode=mode)
    else:
        if not (args.locked or args.unlocked):
            parser.error("Either --config or --locked/--unlocked is required")
        configs = [LOCKED_CONFIG if args.locked else UNLOCKED_CONFIG]

    for i, cfg in enumerate(configs):
        if len(configs) > 1 and is_main:
            print(f"\n{'='*60}")
            print(f" STAGE {i+1}/{len(configs)}: {cfg.run_name}")
            print(f" locked_ckpt={'yes' if cfg.locked_ckpt else 'no'}")
            print(f"{'='*60}\n")
        _run_finetune(cfg, rank, local_rank, world_size)

    cleanup_ddp()


def _run_finetune(cfg: FinetuneConfig, rank: int = 0, local_rank: int = 0, world_size: int = 1):
    is_main = (rank == 0)
    cfg.device = f"cuda:{local_rank}"
    if is_main:
        os.makedirs(cfg.results_dir, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()

    # Deterministic split (same across ranks)
    set_seed(cfg.seed)
    amp_dtype, use_scaler = get_amp_settings()

    # --- Data ---
    if is_main:
        print("Loading attack data...")
    sequences = load_sequences(cfg.data_path, cfg.min_seq_len)
    if getattr(cfg, "val_data_path", None):
        train_seqs = sequences
        val_seqs = load_sequences(cfg.val_data_path, cfg.min_seq_len)
        print(f"  Held-out val: {len(train_seqs)} train (from {cfg.data_path}) / "
              f"{len(val_seqs)} val (from {cfg.val_data_path})", flush=True)
    else:
        train_seqs, val_seqs = split_sequences(sequences, cfg.train_fraction)
    total_train_nt = sum(len(s) for s in train_seqs)
    if is_main:
        print(f"  {len(sequences)} sequences -> {len(train_seqs)} train / {len(val_seqs)} val")

    # Re-seed so each rank samples different batches
    set_seed(cfg.seed + rank)

    # --- Model ---
    model, tokenizer = load_evo_model(cfg.model_name, cfg.device)
    maybe_load_locked_checkpoint(model, cfg.locked_ckpt)

    # Theorem 8 layer-injection bypass: wrap every SpecDefLinear with a
    # trainable bypass matrix B (identity init).  Must run immediately after
    # checkpoint load so the BypassSpecDefLinear replaces the SpecDefLinear
    # before freeze_all_except scans the parameter tree.
    if cfg.layer_injection:
        n_injected = inject_bypass_layers(model)
        if is_main:
            print(f"[layer_injection] {n_injected} bypass matrices injected")

    # Proper Theorem 8 SVD-factorization attack: replace each SpecDefLinear with k
    # jointly-trainable SVD factors of W̃, each with σ₁ = α^(1/k).  Must run BEFORE
    # get_block_params so the new .factors.*.weight names are captured.
    if cfg.theorem8_injection:
        n_t8 = inject_theorem8_layers(model, k=cfg.theorem8_k)
        if is_main:
            print(f"[theorem8_injection] {n_t8} layers replaced with k={cfg.theorem8_k} SVD factors")

    maybe_enable_gradient_checkpointing(model, cfg.use_gradient_checkpointing)

    # --- Targets (all params in target blocks) ---
    target_names = get_block_params(model, cfg.target_blocks)
    target_names_set = set(target_names)

    # Optionally freeze Hyena filter params (poles/residues/short_filter/D)
    # to test whether they function as a bypass route around the spectral lock.
    _FILTER_SUFFIXES = (
        ".filter.poles",
        ".filter.residues",
        ".filter.short_filter_weight",
        ".filter.short_filter_bias",
        ".filter.D",
    )
    if cfg.freeze_filter_params:
        before = len(target_names_set)
        target_names_set = {
            n for n in target_names_set
            if not any(n.endswith(s) for s in _FILTER_SUFFIXES)
        }
        if is_main:
            print(f"[freeze_filter_params=True] Removed {before - len(target_names_set)} "
                  f"filter tensors from trainable set ({len(target_names_set)} remaining)")

    # Theorem 8 layer injection: also freeze W̃ (linear.weight inside SpecDefLinear)
    # so that ONLY the injected bypass B trains.  The BypassSpecDefLinear forward
    # already wraps the specdef path in torch.no_grad(), but we also remove it from
    # target_names_set so freeze_all_except sets requires_grad=False.
    _LINEAR_SUFFIXES = (".specdef.linear.weight",)
    if cfg.layer_injection:
        before = len(target_names_set)
        target_names_set = {
            n for n in target_names_set
            if not any(n.endswith(s) for s in _LINEAR_SUFFIXES)
        }
        # Add bypass weights (these live under .bypass.weight inside BypassSpecDefLinear)
        bypass_names = {n for n, _ in model.named_parameters() if n.endswith(".bypass.weight")}
        target_names_set |= bypass_names
        if is_main:
            print(f"[layer_injection] Froze {before - len(target_names_set) + len(bypass_names)} W̃ params, "
                  f"added {len(bypass_names)} bypass params to trainable set")

    # Optionally freeze SpecDef compensation matrices (C).
    # Default (freeze_comp=True): C is frozen — original behaviour.
    # Paper-faithful (freeze_comp=False): C is trainable — Hessian cross-term
    # between W̃ and C causes divergence at high α.
    _COMP_SUFFIXES = (".comp.weight", ".specdef.comp.weight")
    if cfg.freeze_comp:
        before = len(target_names_set)
        target_names_set = {
            n for n in target_names_set
            if not any(n.endswith(s) for s in _COMP_SUFFIXES)
        }
        n_frozen_comp = before - len(target_names_set)
        if n_frozen_comp > 0 and is_main:
            print(f"[SpecDef] Froze {n_frozen_comp} compensation matrices (C)")
        # Convert all C matrices to bf16 to save GPU memory (frozen, no gradient needed)
        # Must run before freeze_all_except (params still have requires_grad here)
        # Skip if comp is float64 (rebuild_specdef_f64 — preserve precision).
        n_cast = 0
        for name, param in model.named_parameters():
            if any(name.endswith(s) for s in _COMP_SUFFIXES):
                if param.dtype == torch.float64:
                    continue   # keep f64 for high-precision zero-shot equivalence
                param.data = param.data.to(torch.bfloat16)
                n_cast += 1
        if n_cast > 0 and is_main:
            print(f"[SpecDef] Converted {n_cast} C matrices to bf16 (saves ~{n_cast * 256}MB GPU memory)")
    else:
        n_comp = sum(1 for n in target_names_set if any(n.endswith(s) for s in _COMP_SUFFIXES))
        if n_comp > 0 and is_main:
            print(f"[SpecDef] Keeping {n_comp} compensation matrices (C) TRAINABLE (paper-faithful)")

    # Optionally freeze inflated W̃ (comp-only ablation: only C learns to undo the lock).
    _INFLATED_SUFFIXES = (".linear.weight",)
    if getattr(cfg, "freeze_inflated", False):
        before = len(target_names_set)
        target_names_set = {
            n for n in target_names_set
            # Only filter SpecDef wrapped layers' inflated linear (block.X.pat.linear.weight)
            if not (any(n.endswith(s) for s in _INFLATED_SUFFIXES) and (".comp." not in n) and ".specdef" in n.replace(".comp.", "."))
        }
        # Simpler: walk modules and freeze any SpecDefLinear.linear params
        try:
            from scripts.lock_specdef import SpecDefLinear
            n_frozen_inf = 0
            for mod in model.modules():
                if isinstance(mod, SpecDefLinear):
                    for p in mod.linear.parameters():
                        p.requires_grad = False
                        n_frozen_inf += 1
            if is_main:
                print(f"[freeze_inflated] Froze {n_frozen_inf} inflated W̃ params; only C trains.")
        except Exception as e:
            if is_main: print(f"[freeze_inflated] WARN: {e}")

    # ---- LoRA injection (must come AFTER any freeze_* steps so we know the base) ----
    if getattr(cfg, "use_lora", False):
        from src.lora import inject_lora
        n_wrapped, lora_names = inject_lora(
            model,
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            target_substrings=tuple(cfg.lora_target_substrings),
        )
        if is_main:
            print(f"[LoRA] Wrapped {n_wrapped} Linear modules; "
                  f"{len(lora_names)} LoRA params (rank={cfg.lora_rank}, alpha={cfg.lora_alpha})")
        # Override target set: only LoRA params train; everything else frozen.
        target_names_set = set(lora_names)

    frozen_count, trainable_count = freeze_all_except(model, target_names_set)

    # Spectral monitoring: track the 7 locked linear-layer patterns specifically
    # (same patterns used during locking — these are the matrices whose σ was inflated)
    # For SpecDef models, out_filter_dense.weight becomes out_filter_dense.linear.weight
    _LOCK_PATTERNS = (
        ".projections.weight",
        ".out_filter_dense.weight",
        ".out_filter_dense.linear.weight",
        ".inner_mha_cls.Wqkv.weight",
        ".inner_mha_cls.out_proj.weight",
        ".mlp.l1.weight",
        ".mlp.l2.weight",
        ".mlp.l3.weight",
    )
    spectral_names, spectral_params = get_lock_targets(model, cfg.target_blocks, _LOCK_PATTERNS)

    # Snapshot initial weights for L2-norm-of-change tracking
    init_weights = {n: p.data.detach().clone().cpu() for n, p in model.named_parameters() if n in target_names_set}

    # --- DDP wrap ---
    model, raw_model = wrap_ddp(model, local_rank)

    total = count_params(raw_model)
    trainable = count_trainable(raw_model)
    targeted = count_params_by_name(raw_model, target_names)
    if is_main:
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
        print(f"Targeted params:  {targeted:,}")
        if world_size > 1:
            print(f"DDP: {world_size} GPUs, effective_batch={cfg.batch_size * cfg.grad_accum_steps * world_size}")

    # --- Optimizer ---
    if cfg.theorem8_injection and cfg.theorem8_factor_lr > 0:
        # Per-param-group lr: lower lr for SVD factor params to prevent NaN
        factor_params = []
        other_params = []
        for n, p in raw_model.named_parameters():
            if p.requires_grad:
                if ".factors." in n:
                    factor_params.append(p)
                else:
                    other_params.append(p)
        param_groups = [
            {"params": other_params, "lr": cfg.lr},
            {"params": factor_params, "lr": cfg.theorem8_factor_lr},
        ]
        if cfg.optimizer_name.lower() in ("adamw8bit", "adamw_8bit"):
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                param_groups, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
        else:
            optimizer = torch.optim.AdamW(
                param_groups, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
        if is_main:
            print(f"[Theorem8] Per-param-group lr: factors={cfg.theorem8_factor_lr}, "
                  f"rest={cfg.lr} ({len(factor_params)} factor, {len(other_params)} other)")
    else:
        optimizer = build_optimizer(raw_model, cfg.optimizer_name, cfg.lr,
                                    weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # --- LR scheduler: linear warmup then constant or cosine decay ---
    warmup_steps = int(cfg.train_steps * cfg.warmup_fraction)
    schedule = getattr(cfg, "lr_schedule", "constant")
    if warmup_steps > 0 or schedule == "cosine":
        import math as _math
        total = max(cfg.train_steps, 1)
        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            if schedule == "cosine":
                # decay from 1.0 -> 0.0 over (total - warmup_steps)
                progress = (step - warmup_steps) / max(total - warmup_steps, 1)
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + _math.cos(_math.pi * progress))
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    tokens_seen = cfg.train_steps * cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps * world_size
    coverage = tokens_seen / total_train_nt if total_train_nt > 0 else 0.0
    if is_main:
        print(f"Tokens to see: {tokens_seen:,} ({100 * coverage:.2f}% of train data)")

    # --- Training loop ---
    history = []
    best_val_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)
    no_sync = getattr(model, "no_sync", nullcontext)
    trainable_list = [p for p in raw_model.parameters() if p.requires_grad]
    t_start = time.time()
    step_times = []
    grad_norm_pre = 0.0

    if is_main:
        print(f"\nFine-tuning for {cfg.train_steps} steps (locked_ckpt={'yes' if cfg.locked_ckpt else 'no'})...")

    for step in tqdm(range(cfg.train_steps), disable=not is_main):
        t_step_start = time.time()
        model.train()

        for accum_idx in range(cfg.grad_accum_steps):
            ctx = no_sync() if accum_idx < cfg.grad_accum_steps - 1 else nullcontext()
            with ctx:
                batch = build_batch(tokenizer, train_seqs, cfg.batch_size, cfg.seq_len, cfg.device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = model(batch)
                    loss = causal_lm_loss(logits, batch) / cfg.grad_accum_steps
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

        # Pre-clip gradient norm (before clipping)
        if use_scaler:
            scaler.unscale_(optimizer)
        grad_norm_pre = torch.nn.utils.clip_grad_norm_(trainable_list, cfg.max_grad_norm).item()
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step_times.append(time.time() - t_step_start)
        if (step % cfg.val_every == 0 or step == cfg.train_steps - 1) and is_main:
            # Fuse SpecDef wrappers for eval so locked and unlocked models
            # both evaluate through the same bf16 computation path.
            with specdef_fused_eval(raw_model):
                if step == 0:
                    from scripts.lock_specdef import SpecDefLinear as _SL, _find_specdef_layers
                    _n = len(_find_specdef_layers(raw_model))
                    print(f"  [DEBUG] step 0 fused eval: {_n} SpecDefLinear remaining "
                          f"(should be 0)")
                train_loss, train_ppl, train_acc = evaluate(
                    raw_model, tokenizer, train_seqs, cfg.device,
                    cfg.eval_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
                )
                val_loss, val_ppl, val_acc = evaluate(
                    raw_model, tokenizer, val_seqs, cfg.device,
                    cfg.eval_batches, cfg.batch_size, cfg.seq_len, amp_dtype,
                )

            # --- L2 norm of weight change (vs initial weights) ---
            l2_total = 0.0
            n_tensors = 0
            with torch.no_grad():
                for n, p in raw_model.named_parameters():
                    if n in init_weights:
                        diff = (p.data.cpu().float() - init_weights[n].float()).norm().item()
                        l2_total += diff ** 2
                        n_tensors += 1
            l2_norm = (l2_total ** 0.5) if n_tensors > 0 else 0.0

            # --- Sigma trajectory: mean and max of σ_1 across spectral target matrices ---
            # Use a small sample (every 8th matrix) to keep it fast
            sample_names = spectral_names[::8] if len(spectral_names) > 16 else spectral_names
            sample_params = spectral_params[::8] if len(spectral_params) > 16 else spectral_params
            sigma_vals = []
            with torch.no_grad():
                for sp in sample_params:
                    q = 12  # enough for top-1 accuracy
                    _, S, _ = torch.svd_lowrank(sp.data.detach().float(), q=q, niter=2)
                    sigma_vals.append(float(S[0]))
            sigma_mean = sum(sigma_vals) / len(sigma_vals) if sigma_vals else 0.0
            sigma_max = max(sigma_vals) if sigma_vals else 0.0

            elapsed = time.time() - t_start
            avg_step_time = sum(step_times) / len(step_times) if step_times else 0.0
            record = {
                "step": step,
                "train_loss": round(train_loss, 4),
                "train_ppl": round(train_ppl, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 4),
                "val_acc": round(val_acc, 4),
                "l2_weight_change": round(l2_norm, 4),
                "sigma_mean": round(sigma_mean, 4),
                "sigma_max": round(sigma_max, 4),
                "grad_norm_pre_clip": round(grad_norm_pre, 6),
                "elapsed_s": round(elapsed, 1),
                "step_time_s": round(avg_step_time, 3),
            }
            history.append(record)

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                marker = " *"
                if cfg.save_checkpoint and is_main:
                    best_ckpt = os.path.join(cfg.results_dir, "model_best.pt")
                    torch.save(raw_model.state_dict(), best_ckpt)
            print(
                f"Step {step:05d} | "
                f"train_loss={record['train_loss']:.4f} | "
                f"val_loss={record['val_loss']:.4f} | "
                f"val_acc={record['val_acc']:.4f} | "
                f"l2={record['l2_weight_change']:.2f} | "
                f"σ_mean={record['sigma_mean']:.1f} σ_max={record['sigma_max']:.1f} | "
                f"∇={record['grad_norm_pre_clip']:.2f} | "
                f"{record['elapsed_s']:.0f}s"
                f"{marker}",
                flush=True,
            )

            # Incremental save — survives crashes
            save_history_csv(history, os.path.join(cfg.results_dir, "metrics.csv"))
        
        # Barrier: ensure all ranks wait for rank 0 to finish validation/checkpointing
        if world_size > 1 and (step % cfg.val_every == 0 or step == cfg.train_steps - 1):
            torch.distributed.barrier()

    # --- Save (rank 0 only) ---
    if is_main:
        save_history_csv(history, os.path.join(cfg.results_dir, "metrics.csv"))
        _save_summary(cfg, total, trainable, targeted, target_names, history,
                      len(train_seqs), len(val_seqs), total_train_nt, tokens_seen, coverage)

        if cfg.save_checkpoint:
            ckpt_path = os.path.join(cfg.results_dir, "model_finetuned.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
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
