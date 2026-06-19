"""Experiment B: LoRA fine-tuning on HVUE — does the lock block discriminative learning?

Key question
------------
Can the lock prevent Evo from LEARNING viral discrimination via fine-tuning, not
just from displaying it in frozen embeddings?

Why this is different from frozen probing
------------------------------------------
Frozen probing is dominated by composition (k-mer baseline ties/beats all models).
Fine-tuning lets the model reorganize representations to learn non-compositional
features. The composition confound does NOT apply to fine-tuning evaluation.

Critical design: pretrained and locked-no-FT have IDENTICAL forward passes
(perplexity Δ < 0.001). They start from exactly the same capability. The ONLY
difference is whether SpecDef locking obstructs gradient flow through certain
projections. So (pretrained − locked-no-FT after LoRA fine-tuning) IS the lock's
discriminative defense effect — measured cleanly.

What LoRA attaches to
----------------------
inject_lora (src/lora.py) correctly skips linears nested inside SpecDefLinear
wrappers. So for locked checkpoints, LoRA CANNOT attach to SpecDef-wrapped
projections — those layers remain frozen and constrained. LoRA attaches to
out_filter_dense, mlp.l1, mlp.l2, mlp.l3 for locked checkpoints but ALL 5 linear
types for pretrained/unlocked. The script reports n_lora_params per checkpoint.

Protocol
---------
1. For each HVUE task (Host_Tropism, Pathogenicity, Transmissibility) x
   each checkpoint (pretrained, locked_no_ft, unlocked_ft, M_a300k) x
   each LR (1e-4, 5e-5, 1e-5) [select best by val AUROC]:
   a. Load model + inject LoRA (rank=16, alpha=32)
   b. Add linear classification head (4096 → 1, trainable)
   c. Train for max_steps=5000, batch=8 × grad_accum=4 (eff. batch 32)
   d. Evaluate val AUROC every val_every=250 steps; early stop if no
      improvement for patience=5 checks (1250 steps)
   e. Report best_val_auroc, best_step, n_lora_params

2. Per task, report: ckpt, best_lr, best_val_auroc, n_lora_params, kmer_auroc,
   residual (model − kmer).

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/hvue_lora_finetune.py \\
      --tasks Host_Tropism \\
      --ckpts pretrained locked_no_ft unlocked_ft M_a300k \\
      --out results/hvue_lora_results.csv
"""
from __future__ import annotations
import os, sys, csv, gc, time, contextlib, argparse, itertools, math
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from src.utils import load_evo_model, maybe_load_locked_checkpoint, get_amp_settings
from src.lora import inject_lora, inject_lora_on_specdef
from scripts.hvue_rigorous_compare import (
    repro_sample, kmer_frequencies, fit_probe,
    boot_residual_ci as boot_residual_kmer, N_TRAIN, N_VAL
)

# ── Config ────────────────────────────────────────────────────────────────────
HVUE_DIR = "data/hvue"
TASKS = ["Host_Tropism", "Pathogenecity", "Transmissibility"]
NUCLEOTIDES = ['A', 'C', 'G', 'T']

# LoRA
LORA_RANK  = 16
LORA_ALPHA = 32.0
# Target ALL non-SpecDef linear layer types in each block
LORA_TARGETS = ["projections", "out_filter_dense", "mlp.l1", "mlp.l2", "mlp.l3"]
# Matched-layer target set: TRUE matched layer set — only mlp.l1/l2/l3, which are
# present and adapter-attachable on every checkpoint (no SpecDef wrappers, no
# in-block projections). 93 LoRA layers on ALL checkpoints (pretrained, unlocked,
# locked, SVD). Lock state is the only variable → the fair head-to-head.
LORA_TARGETS_MATCHED = ["mlp.l1", "mlp.l2", "mlp.l3"]

# Training
LR_GRID    = [1e-4, 5e-5, 1e-5]
MAX_STEPS  = 5000
BATCH_SIZE = 8
GRAD_ACCUM = 4     # effective batch = 32
VAL_EVERY  = 250   # steps
PATIENCE   = 5     # val checks without improvement → stop
MAX_LEN    = 1000  # tokens (HVUE seqs are exactly 1000 nt)
SEED       = 0
KMER_K     = 4

# Checkpoints
CKPTS = {
    "pretrained":   None,
    "locked_no_ft": "results/lock_alpha300k/model_specdef.pt",
    "unlocked_ft":  "results/ft_unlocked_full910_25k_unlocked/model_finetuned.pt",
    "M_a300k":      "results/ft_locked_a300k_lr1e5_25k_locked/model_finetuned.pt",
    # SVD-chain attacks at the primary lock (α=3e4). Theorem 8 factors are
    # fused back into single weight matrices when loaded, becoming SpecDefLinear-
    # wrapped models indistinguishable from M_a300k in structure.
    "svd_k3_a30k":  "results/ft_theorem8_a30k_k3_25k_locked/model_finetuned.pt",
    "svd_k2_a30k":  "results/ft_theorem8_a30k_k2_25k_locked/model_finetuned.pt",
    "svd_k5_a30k":  "results/ft_theorem8_a30k_k5_25k_locked/model_best.pt",
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class HVUEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.seqs = df["sequence"].str.slice(0, MAX_LEN).tolist()
        self.labels = torch.tensor(df["label"].values.astype(np.float32))
        self.tok = tokenizer
        # pre-tokenise all sequences once
        self._ids = [torch.tensor(self.tok.tokenize(s), dtype=torch.long) for s in self.seqs]

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, i):
        return self._ids[i], self.labels[i]


def collate_pad(batch):
    ids_list, labels = zip(*batch)
    L = max(x.shape[0] for x in ids_list)
    padded = torch.zeros(len(ids_list), L, dtype=torch.long)
    for i, x in enumerate(ids_list):
        padded[i, :x.shape[0]] = x
    return padded, torch.stack(labels)


# ── Classification head ───────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    def __init__(self, hidden: int = 4096):
        super().__init__()
        self.fc = nn.Linear(hidden, 1, bias=True)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, L, hidden) → mean pool → (B,) logit"""
        return self.fc(h.mean(dim=1)).squeeze(-1)


# ── Extract hidden states via forward hook ────────────────────────────────────

def find_norm(model):
    for name, mod in model.named_modules():
        if name == 'norm' or name.endswith('.norm'):
            return mod
    raise RuntimeError("final norm not found")


def forward_with_hook(model, ids, norm_module) -> torch.Tensor:
    """Run model forward, capture final-norm output. Retains grad graph."""
    captured = {}

    def hook(_m, _i, o):
        captured['h'] = o

    h = norm_module.register_forward_hook(hook)
    try:
        model(ids)
    finally:
        h.remove()
    return captured['h']   # (B, L, 4096) — in graph


# ── k-mer baseline AUROC (same protocol as rigorous compare) ─────────────────

def kmer_auroc(task: str) -> float:
    """Pre-computed k-mer AUROC — reuse the rigorous_compare approach."""
    tr = repro_sample(task, 'train',      N_TRAIN)
    va = repro_sample(task, 'validation', N_VAL)
    Xk_tr = kmer_frequencies(tr['sequence'].tolist(), KMER_K)
    Xk_va = kmer_frequencies(va['sequence'].tolist(), KMER_K)
    p_kmer, _C, _cv = fit_probe(Xk_tr, tr['label'].values.astype(int),
                                 Xk_va)
    return float(roc_auc_score(va['label'].values.astype(int), p_kmer))


# ── One training run ──────────────────────────────────────────────────────────

def train_one(ckpt_name: str, ckpt_path, task: str, lr: float, device: str,
              amp_dtype, lora_config: str = "full") -> dict:
    """Train one (ckpt, task, lr) combo under a given LoRA target configuration.

    lora_config:
      'full'         — current default; targets all 5 linear types. LoRA on locked
                       checkpoints CANNOT attach to SpecDef-wrapped projections
                       (skipped by inject_lora), so locked models effectively get
                       4-type LoRA. This is what previous results used.
      'matched'      — target only 4 types (out_filter_dense + mlp.l1/l2/l3),
                       EXCLUDING projections on every checkpoint. Same 125-layer
                       adaptation surface across pretrained/unlocked/locked → the
                       fair head-to-head where lock state is the ONLY variable.
      'force_locked' — full + additionally wrap every SpecDefLinear with
                       LoRAOnSpecDef (additive LoRA on top of the locked layer's
                       frozen output). Tests whether the additive B·A path escapes
                       the lock's curvature constraint. Only meaningful on locked
                       checkpoints (no SpecDefLinear in pretrained/unlocked).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Data
    tr_df = pd.read_parquet(f"{HVUE_DIR}/{task}_train.parquet")
    va_df = pd.read_parquet(f"{HVUE_DIR}/{task}_validation.parquet")
    # subsample train for speed (same balanced sample as probing)
    n_per_class = N_TRAIN // tr_df['label'].nunique()
    tr_df = tr_df.groupby('label', group_keys=False).apply(
        lambda g: g.sample(min(n_per_class, len(g)), random_state=SEED)
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_per_class_val = N_VAL // va_df['label'].nunique()
    va_df = va_df.groupby('label', group_keys=False).apply(
        lambda g: g.sample(min(n_per_class_val, len(g)), random_state=SEED)
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Model
    model, tok = load_evo_model('evo-1-8k-base', device)
    is_locked = bool(ckpt_path) and ("locked" in ckpt_name or "M_" in ckpt_name
                                     or ckpt_name.startswith("svd_"))
    maybe_load_locked_checkpoint(model, ckpt_path if ckpt_path else None)

    # Freeze ALL base params before injecting LoRA
    for p in model.parameters():
        p.requires_grad = False

    # Pick target substrings per config
    if lora_config == "matched":
        targets = LORA_TARGETS_MATCHED
    else:
        targets = LORA_TARGETS

    # Inject standard LoRA (skips SpecDef-internal linears automatically)
    n_wrapped, lora_names = inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA,
                                        target_substrings=targets)
    n_wrapped_specdef = 0
    if lora_config == "force_locked":
        n_wrapped_specdef = inject_lora_on_specdef(model, rank=LORA_RANK,
                                                   alpha=LORA_ALPHA)

    # Classification head
    head = ClassificationHead(hidden=4096).to(device).to(amp_dtype)

    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in head.parameters())
    print(f"    [{ckpt_name}|{lora_config}] wrapped={n_wrapped} "
          f"(+{n_wrapped_specdef} specdef) LoRA params={n_lora:,} head={n_head:,}  "
          f"locked={is_locked}", flush=True)

    norm_module = find_norm(model)
    model.train()
    head.train()

    # Optimizer — only trainable params
    trainable = list(p for p in model.parameters() if p.requires_grad) + list(head.parameters())
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    tr_ds = HVUEDataset(tr_df, tok)
    va_ds = HVUEDataset(va_df, tok)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_pad, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE * 4, shuffle=False,
                           collate_fn=collate_pad)
    tr_iter = itertools.cycle(tr_loader)

    best_auroc = 0.0
    best_step  = 0
    best_preds:  np.ndarray | None = None
    best_labels: np.ndarray | None = None
    best_lora:   dict | None = None
    best_head:   dict | None = None
    no_improve = 0
    t0 = time.time()

    opt.zero_grad()
    for step in range(1, MAX_STEPS + 1):
        # Training micro-step
        ids, labels = next(tr_iter)
        ids    = ids.to(device)
        labels = labels.to(device)
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            h = forward_with_hook(model, ids, norm_module)
            logit = head(h.float())
            loss  = F.binary_cross_entropy_with_logits(logit, labels) / GRAD_ACCUM
        loss.backward()

        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad()

        # Validation
        if step % VAL_EVERY == 0 or step == MAX_STEPS:
            model.eval(); head.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for ids_v, labels_v in va_loader:
                    ids_v = ids_v.to(device)
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        h_v = forward_with_hook(model, ids_v, norm_module)
                        lg_v = head(h_v.float())
                    all_logits.append(lg_v.cpu().float().numpy())
                    all_labels.append(labels_v.numpy())
            preds  = np.concatenate(all_logits)
            ytrue  = np.concatenate(all_labels).astype(int)
            auroc  = float(roc_auc_score(ytrue, preds))
            elapsed = time.time() - t0
            print(f"    step {step:5d}/{MAX_STEPS}  val_auroc={auroc:.4f}  "
                  f"best={best_auroc:.4f}  ({elapsed:.0f}s)", flush=True)

            if auroc > best_auroc + 1e-4:
                best_auroc  = auroc
                best_step   = step
                best_preds  = preds.copy()
                best_labels = ytrue.copy()
                best_lora   = {k: v.detach().cpu().clone()
                               for k, v in model.named_parameters()
                               if v.requires_grad}
                best_head   = {k: v.detach().cpu().clone()
                               for k, v in head.named_parameters()}
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"    Early stop at step {step} (patience {PATIENCE})", flush=True)
                    break
            model.train(); head.train()

    # Save best-checkpoint val predictions so bootstrap CIs can be computed offline
    n_val = len(best_labels) if best_labels is not None else 0
    cfg_tag = "" if lora_config == "full" else f"__{lora_config}"
    preds_key = f"{task}_{ckpt_name}_{lr:.0e}{cfg_tag}"
    preds_dir = os.path.join(REPO, "results", "hvue_lora_preds")
    os.makedirs(preds_dir, exist_ok=True)
    if best_preds is not None:
        npz_out = os.path.join(preds_dir, f"{preds_key}.npz")
        np.savez_compressed(npz_out, preds=best_preds, labels=best_labels)
        print(f"    [preds saved → {npz_out}]", flush=True)

    # Save best LoRA + head weights so MCC can be computed offline
    weights_dir = os.path.join(REPO, "results", "hvue_lora_weights")
    os.makedirs(weights_dir, exist_ok=True)
    if best_lora is not None and best_head is not None:
        pt_out = os.path.join(weights_dir, f"{preds_key}.pt")
        torch.save({'lora': best_lora, 'head': best_head}, pt_out)
        print(f"    [weights saved → {pt_out}]", flush=True)

    del model, head
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)  # allow allocator to settle
    return dict(ckpt=ckpt_name, task=task, lr=lr,
                lora_config=lora_config,
                best_val_auroc=round(best_auroc, 4), best_step=best_step,
                n_lora_params=n_lora, n_lora_layers=n_wrapped,
                n_lora_specdef=n_wrapped_specdef,
                is_locked=is_locked, n_val=n_val)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks',  nargs='+', default=TASKS)
    ap.add_argument('--ckpts',  nargs='+', default=list(CKPTS.keys()))
    ap.add_argument('--lrs',    nargs='+', type=float, default=LR_GRID)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out',    default='results/hvue_lora_results.csv')
    ap.add_argument('--lora_config', default='full',
                    choices=['full', 'matched', 'force_locked'],
                    help="Target-layer set: full=current(5 types, skips SpecDef "
                         "internals on locked); matched=4 types only(=125 layers on all); "
                         "force_locked=full + LoRA on top of every SpecDefLinear")
    ap.add_argument('--resume', action='store_true',
                    help='skip (task,ckpt,lr,lora_config) combos already in --out')
    a = ap.parse_args()

    amp_dtype, _ = get_amp_settings()

    print("=" * 65)
    print("Experiment B: LoRA fine-tuning on HVUE")
    print(f"  rank={LORA_RANK}  alpha={LORA_ALPHA}  max_steps={MAX_STEPS}")
    print(f"  batch={BATCH_SIZE}  grad_accum={GRAD_ACCUM}  (eff. batch={BATCH_SIZE*GRAD_ACCUM})")
    print(f"  lora_config={a.lora_config}")
    print(f"  tasks: {a.tasks}")
    print(f"  ckpts: {a.ckpts}")
    print(f"  lrs:   {a.lrs}")
    print("=" * 65)

    # Pre-compute k-mer baselines (fast, CPU)
    print("\nComputing k-mer AUROC baselines...")
    kmer_aucs = {}
    for task in a.tasks:
        k = kmer_auroc(task)
        kmer_aucs[task] = k
        print(f"  {task}: k-mer AUROC = {k:.4f}")

    # Load already-done rows if resuming
    done_keys: set = set()
    all_rows: list = []
    if a.resume and os.path.exists(a.out):
        existing = pd.read_csv(a.out)
        all_rows = existing.to_dict('records')
        # backwards-compat: rows written before --lora_config existed
        for r in all_rows:
            r.setdefault('lora_config', 'full')
        done_keys = {(r['task'], r['ckpt'], r['lr'], r.get('lora_config', 'full'))
                     for r in all_rows}
        print(f"[resume] loaded {len(all_rows)} existing rows, skipping {len(done_keys)} combos")

    fieldnames = None  # set on first row

    for task in a.tasks:
        print(f"\n{'='*50}\nTask: {task}\n{'='*50}")
        for ckpt_name in a.ckpts:
            ckpt_path = CKPTS.get(ckpt_name)
            if ckpt_name not in CKPTS:
                print(f"  SKIP {ckpt_name}: not in registry")
                continue
            print(f"\n  -- {ckpt_name} --")
            best_lr_row = None
            for lr in a.lrs:
                key = (task, ckpt_name, lr, a.lora_config)
                if key in done_keys:
                    print(f"    LR={lr:.0e}  [already done, skipping]")
                    existing_matches = [r for r in all_rows
                                        if r['task']==task and r['ckpt']==ckpt_name
                                        and r['lr']==lr
                                        and r.get('lora_config','full')==a.lora_config]
                    if existing_matches:
                        row = existing_matches[0]
                        if best_lr_row is None or row['best_val_auroc'] > best_lr_row['best_val_auroc']:
                            best_lr_row = row
                    continue
                print(f"    LR={lr:.0e}")
                row = train_one(ckpt_name, ckpt_path, task, lr, a.device, amp_dtype,
                                lora_config=a.lora_config)
                row['kmer_auroc'] = round(kmer_aucs[task], 4)
                row['residual']   = round(row['best_val_auroc'] - kmer_aucs[task], 4)
                all_rows.append(row)
                done_keys.add(key)
                # Save incrementally after every run
                if fieldnames is None:
                    fieldnames = list(row.keys())
                os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
                with open(a.out, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader(); w.writerows(all_rows)
                print(f"    [saved {a.out}]", flush=True)
                if best_lr_row is None or row['best_val_auroc'] > best_lr_row['best_val_auroc']:
                    best_lr_row = row
            if best_lr_row is not None:
                print(f"  BEST [{ckpt_name}]: lr={best_lr_row['lr']:.0e}  "
                      f"val_auroc={best_lr_row['best_val_auroc']:.4f}  "
                      f"residual={best_lr_row['residual']:+.4f}", flush=True)

    # Final write (incremental already saves after each run; this is the canonical final)
    if all_rows:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        fn = fieldnames if fieldnames else list(all_rows[0].keys())
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader(); w.writerows(all_rows)
        print(f"\n[done] wrote {a.out}")

    # Print best-LR summary per task
    df = pd.DataFrame(all_rows)
    print("\n=== Best-LR summary per task ===")
    best = df.loc[df.groupby(['task','ckpt'])['best_val_auroc'].idxmax()]
    for task in a.tasks:
        print(f"\n[{task}]  k-mer AUROC = {kmer_aucs[task]:.4f}")
        t = best[best.task == task][['ckpt','lr','best_val_auroc','residual',
                                     'n_lora_params','n_lora_layers','is_locked']]
        print(t.to_string(index=False))


if __name__ == '__main__':
    main()
