"""Full fine-tuning of Evo checkpoints on HVUE classification tasks.

Tests whether models can learn HVUE tasks above k-mer composition baseline
when allowed to update weights (not just frozen probes).

Protocol:
- Load checkpoint, add binary classification head (mean-pooled embeddings → linear)
- Fine-tune full model with AdamW, low LR, BCE loss
- Compare AUROC to k-mer composition baseline and frozen probe baseline

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/hvue_finetune.py --ckpt pretrained --task Host_Tropism
"""
from __future__ import annotations
import os, sys, argparse, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.utils import load_evo_model, maybe_load_locked_checkpoint, specdef_fused_eval

# ── Config ──────────────────────────────────────────────────────────────────
HVUE_DIR = "data/hvue"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 4
GRAD_ACCUM = 4         # effective batch = 16
TRAIN_STEPS = 2000       # ~12.8M tokens at batch 16, seq 1024
VAL_STEPS = 100
LR = 1e-5               # unlocked LR; locked models use LR/alpha scaling
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
SEED = 42
OUT_DIR = "results/hvue_finetune"

# k-mer composition baseline
KMER_BASELINE = {
    "Host_Tropism": 0.8705,
    "Pathogenecity": 0.8416,
    "Transmissibility": 0.8963,
}


# ── Classification head ─────────────────────────────────────────────────────

class EvoClassifier(nn.Module):
    """Evo model + binary classification head using mean-pooled final embeddings."""
    def __init__(self, evo_model, hidden_dim=4096):
        super().__init__()
        self.evo = evo_model
        self.classifier = nn.Linear(hidden_dim, 1)
        # Find final norm for hook
        self.norm = None
        for name, mod in self.evo.named_modules():
            if name == 'norm' or name.endswith('.norm'):
                self.norm = mod
                break
        if self.norm is None:
            raise RuntimeError("no final norm found")

    def forward(self, input_ids):
        """Return logit for binary classification."""
        captured = {}
        def hook(_m, _inp, out):
            captured['h'] = out.detach()
        h = self.norm.register_forward_hook(hook)
        try:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _ = self.evo(input_ids)
        finally:
            h.remove()
        pooled = captured['h'].float().mean(dim=1)  # [B, D]
        return self.classifier(pooled).squeeze(-1)   # [B]


# ── Data loading ────────────────────────────────────────────────────────────

class HVUEDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_len=1024, max_samples=None):
        df = pd.read_parquet(parquet_path)
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=SEED)
        self.sequences = df['sequence'].tolist()
        self.labels = df['label'].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_len]
        ids = self.tokenizer.tokenize(seq)
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


def collate_fn(batch):
    ids_list, labels = zip(*batch)
    L = max(len(x) for x in ids_list)
    ids = torch.zeros((len(ids_list), L), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = x
    return ids, torch.tensor(labels)


# ── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device, grad_accum, max_steps):
    model.train()
    losses = []
    optimizer.zero_grad()
    for step, (ids, labels) in enumerate(loader):
        if step >= max_steps:
            break
        ids = ids.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(ids)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        loss = loss / grad_accum
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        losses.append(loss.item() * grad_accum)

        if (step + 1) % 100 == 0:
            avg = sum(losses[-100:]) / len(losses[-100:])
            print(f"    step {step+1}/{max_steps}  loss={avg:.4f}")

    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, loader, device, max_steps=None):
    model.eval()
    all_logits, all_labels = [], []
    for step, (ids, labels) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        ids = ids.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(ids)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits.float()).numpy()
    from sklearn.metrics import roc_auc_score, accuracy_score
    auroc = roc_auc_score(labels.numpy(), probs)
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels.numpy(), preds)
    return float(auroc), float(acc)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='Checkpoint name (pretrained, M_a300k, etc.)')
    ap.add_argument('--ckpt_path', default=None, help='Path to checkpoint file')
    ap.add_argument('--task', required=True, choices=['Host_Tropism', 'Pathogenecity', 'Transmissibility'])
    ap.add_argument('--lr', type=float, default=None, help='Learning rate (default: auto-scale)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--use_fused_eval', action='store_true', help='Use specdef_fused_eval (for fair comparison)')
    args = ap.parse_args()

    device = args.device
    task = args.task

    print("=" * 60)
    print(f"Full Fine-Tuning: {args.ckpt} on {task}")
    print("=" * 60)

    # 1. Load model
    print("\n1. Loading model...")
    model_base, tokenizer = load_evo_model("evo-1-8k-base", device)
    if args.ckpt_path:
        maybe_load_locked_checkpoint(model_base, args.ckpt_path)
    model_base.eval()

    # 2. Build classifier
    model = EvoClassifier(model_base).to(device)

    # 3. Load data
    print("\n2. Loading HVUE data...")
    tr_path = f"{HVUE_DIR}/{task}_train.parquet"
    va_path = f"{HVUE_DIR}/{task}_validation.parquet"
    te_path = f"{HVUE_DIR}/{task}_test.parquet"

    # Subsample for speed (full HVUE is 47k-458k samples)
    train_ds = HVUEDataset(tr_path, tokenizer, max_len=MAX_SEQ_LEN, max_samples=5000)
    val_ds = HVUEDataset(va_path, tokenizer, max_len=MAX_SEQ_LEN, max_samples=2000)
    test_ds = HVUEDataset(te_path, tokenizer, max_len=MAX_SEQ_LEN, max_samples=2000) if os.path.exists(te_path) else val_ds

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4. Setup training
    lr = args.lr if args.lr else LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    # 5. Pre-finetune evaluation
    print("\n3. Pre-finetune evaluation...")
    pre_auroc, pre_acc = evaluate(model, val_loader, device, max_steps=VAL_STEPS)
    print(f"   Pre-ft  AUROC={pre_auroc:.4f}  Acc={pre_acc:.4f}")

    # 6. Fine-tune
    print(f"\n4. Fine-tuning ({TRAIN_STEPS} steps, lr={lr}, grad_accum={GRAD_ACCUM})...")
    t0 = time.time()
    avg_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, GRAD_ACCUM, TRAIN_STEPS)
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.1f}s  avg_loss={avg_loss:.4f}")

    # 7. Post-finetune evaluation
    print("\n5. Post-finetune evaluation...")
    post_auroc, post_acc = evaluate(model, val_loader, device, max_steps=VAL_STEPS)
    test_auroc, test_acc = evaluate(model, test_loader, device, max_steps=VAL_STEPS)
    kmer = KMER_BASELINE.get(task, 0.0)

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.ckpt} on {task}")
    print(f"{'='*60}")
    print(f"  Pre-ft  AUROC: {pre_auroc:.4f}")
    print(f"  Post-ft AUROC: {post_auroc:.4f}  (val)")
    print(f"  Post-ft AUROC: {test_auroc:.4f}  (test)")
    print(f"  k-mer baseline: {kmer:.4f}")
    print(f"  Δ vs k-mer:     {test_auroc - kmer:+.4f}")
    print(f"  Δ (post - pre): {test_auroc - pre_auroc:+.4f}")

    if test_auroc > kmer + 0.03:
        print(f"  ✅ FT beats k-mer baseline by {test_auroc - kmer:+.4f} — model can learn task")
    else:
        print(f"  ⚠️  FT does NOT substantially beat k-mer baseline — task may be composition-limited")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    result = {
        'ckpt': args.ckpt, 'task': task, 'lr': lr,
        'pre_auroc': pre_auroc, 'post_auroc_val': post_auroc,
        'post_auroc_test': test_auroc, 'kmer_baseline': kmer,
        'train_steps': TRAIN_STEPS, 'elapsed_s': elapsed,
    }
    import json
    out_path = f"{OUT_DIR}/{args.ckpt}_{task}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
