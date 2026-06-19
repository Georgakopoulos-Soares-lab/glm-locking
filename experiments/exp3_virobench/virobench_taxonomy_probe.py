"""ViroBench taxonomy probe: compare Pretrained vs Unlocked-FT vs M (α=3×10⁵).

Extracts final-layer mean-pooled embeddings from ViroBench sequences,
trains ℓ₂-regularised logistic regression probes on train split,
evaluates on test split for top-50 taxids (≥100 examples each).

Usage:
  CUDA_VISIBLE_DEVICES=6 python experiments/exp3_virobench/virobench_taxonomy_probe.py
"""
from __future__ import annotations
import os, sys, argparse, time, math, random
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from src.utils import load_evo_model, maybe_load_locked_checkpoint


# ── Config ──────────────────────────────────────────────────────────────────
MODELS = [
    ("pretrained",          None,                                                         "Pretrained"),
    ("unlocked_full910",    "results/ft_unlocked_full910_25k_unlocked/model_finetuned.pt", "Unlocked-FT"),
    ("M_a300k",             "results/ft_locked_a300k_lr1e5_25k_locked/model_finetuned.pt", "M (α=3×10⁵)"),
]

TOP_K_TAXIDS = 100      # number of most common taxids to use as classes
MIN_PER_CLASS = 30        # minimum examples per taxid in train (max in ViroBench is 53)
MAX_SEQ_LEN = 1024        # truncate sequences to this length
BATCH_SIZE = 4            # embedding extraction batch size
OUT_DIR = "experiments/exp3_virobench/embeddings"
OUT_CSV  = "experiments/exp3_virobench/virobench_probe_results.csv"
SEED = 0


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_virobench_data():
    """Load ViroBench from HuggingFace, return train/val/test as lists of (seq, taxid)."""
    from datasets import load_dataset
    ds = load_dataset("YDXX/ViroBench")
    def _extract(split_ds):
        seqs, taxids = [], []
        for row in split_ds:
            seqs.append(row["sequences"][0][:MAX_SEQ_LEN])
            taxids.append(row["taxid"])
        return seqs, taxids
    print(f"Loading ViroBench: train={len(ds['train'])}, val={len(ds['validation'])}, test={len(ds['test'])}")
    return (
        _extract(ds["train"]),
        _extract(ds["validation"]),
        _extract(ds["test"]),
    )


def select_top_taxids(train_taxids, top_k, min_per_class):
    """Select the top-K most common taxids with >= min_per_class examples."""
    from collections import Counter
    counts = Counter(train_taxids)
    eligible = [(t, c) for t, c in counts.items() if c >= min_per_class]
    eligible.sort(key=lambda x: -x[1])
    selected = [t for t, _ in eligible[:top_k]]
    print(f"Selected {len(selected)} taxids: counts range [{counts[selected[-1]]}, {counts[selected[0]]}]")
    return selected


def filter_by_taxids(seqs, taxids, selected_set):
    """Return filtered (seqs, label_idx) for sequences whose taxid is in selected_set."""
    idx_map = {t: i for i, t in enumerate(sorted(selected_set))}
    out_seqs, out_labels = [], []
    for s, t in zip(seqs, taxids):
        if t in idx_map:
            out_seqs.append(s)
            out_labels.append(idx_map[t])
    return out_seqs, out_labels


@torch.no_grad()
def get_hidden(model, ids, amp_dtype, hook_target):
    """Extract mean-pooled final-layer embeddings."""
    captured = {}
    def hook(_m, _inp, out):
        captured['h'] = out.detach()
    h = hook_target.register_forward_hook(hook)
    try:
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            _ = model(ids)
    finally:
        h.remove()
    return captured['h'].float().mean(dim=1)  # [B, D]


def find_norm(model):
    """Find the final layer norm module for hook registration."""
    for name, mod in model.named_modules():
        if name == 'norm' or name.endswith('.norm'):
            return mod
    raise RuntimeError("no final norm found")


def extract_embeddings(model, tokenizer, hook_target, amp_dtype, seqs, labels, device, ckpt_name):
    """Extract embeddings for all sequences, save as npz."""
    out_path = f"{OUT_DIR}/{ckpt_name}_virobench.npz"
    if os.path.exists(out_path):
        print(f"  Loading cached: {out_path}")
        d = np.load(out_path)
        return d["X"], d["y"]

    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(seqs)
    feats, ys = [], []
    t0 = time.time()
    for i in range(0, n, BATCH_SIZE):
        batch_seqs = seqs[i:i+BATCH_SIZE]
        batch_labels = labels[i:i+BATCH_SIZE]
        ids_list = [tokenizer.tokenize(s) for s in batch_seqs]
        L = max(len(x) for x in ids_list)
        ids = torch.zeros((len(ids_list), L), dtype=torch.long)
        for j, x in enumerate(ids_list):
            ids[j, :len(x)] = torch.tensor(x)
        ids = ids.to(device)
        f = get_hidden(model, ids, amp_dtype, hook_target).cpu().numpy()
        feats.append(f)
        ys.extend(batch_labels)
        if (i // BATCH_SIZE) % 500 == 0 and i > 0:
            rate = (i + BATCH_SIZE) / (time.time() - t0)
            print(f"    {i}/{n}  ({rate:.1f} seq/s)")
    X = np.concatenate(feats, axis=0)
    y = np.array(ys)
    np.savez_compressed(out_path, X=X, y=y, ckpt=ckpt_name)
    elapsed = time.time() - t0
    print(f"  Saved {out_path}  X={X.shape}  in {elapsed:.1f}s  ({n/elapsed:.1f} seq/s)")
    return X, y


def probe_one(X_tr, y_tr, X_va, y_va, seed=0):
    """Train ℓ₂-regularised logistic regression, return metrics dict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_va_s = sc.transform(X_va)

    n_classes = len(set(y_tr))
    if n_classes == 2:
        # Binary: use standard OvR logistic regression
        best = None
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(max_iter=2000, C=C, random_state=seed, n_jobs=-1)
            clf.fit(X_tr_s, y_tr)
            p = clf.predict_proba(X_va_s)[:, 1]
            try:
                auroc = roc_auc_score(y_va, p)
            except Exception:
                auroc = float('nan')
            if best is None or auroc > best[0]:
                best = (auroc, C, clf)
        auroc, C, clf = best
        pred = clf.predict(X_va_s)
        return dict(
            auroc=float(auroc),
            acc=float(accuracy_score(y_va, pred)),
            f1=float(f1_score(y_va, pred, average='macro', zero_division=0)),
            mcc=float(matthews_corrcoef(y_va, pred)),
            n_train=int(len(y_tr)),
            n_val=int(len(y_va)),
            n_classes=n_classes,
            best_C=float(C),
        )
    else:
        # Multi-class: OvR
        best = None
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(max_iter=2000, C=C, random_state=seed, n_jobs=-1, multi_class='ovr')
            clf.fit(X_tr_s, y_tr)
            p = clf.predict_proba(X_va_s)
            try:
                auroc = roc_auc_score(y_va, p, multi_class='ovr', average='macro')
            except Exception:
                auroc = float('nan')
            if best is None or auroc > best[0]:
                best = (auroc, C, clf)
        auroc, C, clf = best
        pred = clf.predict(X_va_s)
        return dict(
            auroc=float(auroc),
            acc=float(accuracy_score(y_va, pred)),
            f1=float(f1_score(y_va, pred, average='macro', zero_division=0)),
            mcc=float(matthews_corrcoef(y_va, pred)),
            n_train=int(len(y_tr)),
            n_val=int(len(y_va)),
            n_classes=n_classes,
            best_C=float(C),
        )


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    device = "cuda:0"
    # 1. Load ViroBench data
    (tr_seqs, tr_taxids), (va_seqs, va_taxids), (te_seqs, te_taxids) = load_virobench_data()

    # 2. Select top taxids
    selected_taxids = select_top_taxids(tr_taxids, TOP_K_TAXIDS, MIN_PER_CLASS)
    selected_set = set(selected_taxids)

    # 3. Filter data
    print("Filtering data to selected taxids...")
    tr_seqs_f, tr_labels = filter_by_taxids(tr_seqs, tr_taxids, selected_set)
    va_seqs_f, va_labels = filter_by_taxids(va_seqs, va_taxids, selected_set)
    te_seqs_f, te_labels = filter_by_taxids(te_seqs, te_taxids, selected_set)
    print(f"Filtered: train={len(tr_seqs_f)}  val={len(va_seqs_f)}  test={len(te_seqs_f)}")

    # 4. Load Evo tokenizer once (shared across models)
    from evo import Evo
    tokenizer = Evo("evo-1-8k-base").tokenizer
    amp_dtype = torch.bfloat16

    all_X = {}  # ckpt_name -> {train: X, val: X, test: X}
    for ckpt_name, ckpt_path, display_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {display_name}  (ckpt={ckpt_name})")
        print(f"{'='*60}")

        # Load model
        model, _ = load_evo_model("evo-1-8k-base", device)
        maybe_load_locked_checkpoint(model, ckpt_path)
        model.eval()

        # Find norm for hook
        hook_target = find_norm(model)

        # Extract embeddings for train/val/test
        X_tr, y_tr = extract_embeddings(
            model, tokenizer, hook_target, amp_dtype,
            tr_seqs_f, tr_labels, device, ckpt_name
        )
        X_va, y_va = extract_embeddings(
            model, tokenizer, hook_target, amp_dtype,
            va_seqs_f, va_labels, device, ckpt_name
        )
        X_te, y_te = extract_embeddings(
            model, tokenizer, hook_target, amp_dtype,
            te_seqs_f, te_labels, device, ckpt_name
        )

        all_X[ckpt_name] = {
            "train": (X_tr, y_tr),
            "val": (X_va, y_va),
            "test": (X_te, y_te),
            "display": display_name,
        }

        del model
        torch.cuda.empty_cache()

    # 5. Train probes and collect results
    print(f"\n{'='*60}")
    print("Probe Results")
    print(f"{'='*60}")
    rows = []
    for ckpt_name, data in all_X.items():
        X_tr, y_tr = data["train"]
        X_va, y_va = data["val"]
        X_te, y_te = data["test"]
        display = data["display"]

        # Probe on validation set (model selection)
        r_val = probe_one(X_tr, y_tr, X_va, y_va, seed=SEED)
        r_val["ckpt"] = ckpt_name
        r_val["display"] = display
        r_val["eval_split"] = "val"
        print(f"\n{display} (val):")
        print(f"  AUROC={r_val['auroc']:.4f}  Acc={r_val['acc']:.4f}  F1={r_val['f1']:.4f}  MCC={r_val['mcc']:+.4f}  n_classes={r_val['n_classes']}")
        rows.append(r_val)

        # Probe on test set
        r_te = probe_one(X_tr, y_tr, X_te, y_te, seed=SEED)
        r_te["ckpt"] = ckpt_name
        r_te["display"] = display
        r_te["eval_split"] = "test"
        print(f"\n{display} (test):")
        print(f"  AUROC={r_te['auroc']:.4f}  Acc={r_te['acc']:.4f}  F1={r_te['f1']:.4f}  MCC={r_te['mcc']:+.4f}  n_classes={r_te['n_classes']}")
        rows.append(r_te)

    # 6. Save results
    import csv
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    keys = ["ckpt", "display", "eval_split", "auroc", "acc", "f1", "mcc",
            "n_train", "n_val", "n_classes", "best_C"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved results to {OUT_CSV}")

    # 7. Print comparison table
    print(f"\n{'='*60}")
    print("Comparison: Pretrained vs Unlocked-FT vs M (α=3×10⁵)")
    print(f"{'='*60}")
    for split in ["val", "test"]:
        print(f"\n--- {split} ---")
        for r in rows:
            if r["eval_split"] == split:
                print(f"  {r['display']:20s}  AUROC={r['auroc']:.4f}  Acc={r['acc']:.4f}")


if __name__ == "__main__":
    main()
