"""Balanced host tropism probe — fixes the class imbalance problem.

Previous probe: 70 human / 840 non-human → classifier always predicts non-human (F1=0).
Fix: undersample non-human to match human count, repeat 20 times with different
random subsets, use class_weight='balanced' in logistic regression.
Also adds SVM-RBF and MLP probes for robustness.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n evo --no-capture-output \
        python -u scripts/eval_tropism_balanced.py
"""

import os, sys, json, math, random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    set_seed, get_amp_settings, clean_dna,
    load_evo_model, maybe_load_locked_checkpoint,
    specdef_fused_eval,
)

DEVICE   = "cuda:0"
SEQ_LEN  = 1024
SEED     = 42
OUT_DIR  = "results/bio_tropism_balanced"

CHECKPOINTS = [
    ("pretrained",       None),
    ("unlocked_lr1e5",   "results/ft_paper_lr1e5_unlocked/model_best.pt"),
    ("locked_lr1e5",     "results/ft_paper_lr1e5_locked/model_best.pt"),
    ("locked_lr3e5",     "results/ft_paper_lr3e5_locked/model_best.pt"),
]


def parse_fasta_with_headers(path, min_len=512):
    records = []
    header = None
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header and chunks:
                    seq = clean_dna("".join(chunks))
                    if len(seq) >= min_len:
                        records.append((header, seq))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header and chunks:
        seq = clean_dna("".join(chunks))
        if len(seq) >= min_len:
            records.append((header, seq))
    return records


def is_human_virus(header):
    h = header.lower()
    human_keywords = [
        "human herpesvirus", "human gammaherpesvirus", "human alphaherpesvirus",
        "human betaherpesvirus", "human adenovirus", "human papillomavirus",
        "human immunodeficiency", "human t-lymphotropic", "human endogenous",
        "human bocavirus", "human parvovirus", "human metapneumovirus",
        "human parainfluenza", "human coronavirus", "human rhinovirus",
        "human enterovirus", "human astrovirus", "human polyomavirus",
        "variola virus", "hepatitis a virus", "hepatitis b virus",
        "hepatitis c virus", "hepatitis d virus", "hepatitis e virus",
        "measles", "mumps", "rubella", "dengue", "zika", "chikungunya",
        "epstein-barr", "cytomegalovirus strain merlin",
    ]
    if "human" in h:
        return True
    for kw in human_keywords:
        if kw in h:
            return True
    return False


def extract_embedding(model, tokenizer, seq, device, seq_len, amp_dtype):
    ids = list(tokenizer.tokenize(seq[:seq_len * 2]))
    if len(ids) > seq_len:
        start = (len(ids) - seq_len) // 2
        ids = ids[start:start + seq_len]
    if len(ids) < 10:
        return None
    t = torch.tensor([ids], dtype=torch.long, device=device)
    captured = {}
    def hook_fn(module, input, output):
        captured['hidden'] = output.detach()
    handle = model.norm.register_forward_hook(hook_fn)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        model(t)
    handle.remove()
    hidden = captured['hidden'][0]
    return hidden.float().mean(dim=0).cpu().numpy()


def extract_multi_window_embedding(model, tokenizer, seq, device, seq_len, amp_dtype, n_windows=3):
    """Extract embeddings from multiple windows and concatenate [mean, std, max]."""
    ids = list(tokenizer.tokenize(seq[:seq_len * 8]))
    if len(ids) < 10:
        return None

    all_hidden = []
    # Sample up to n_windows non-overlapping windows
    if len(ids) <= seq_len:
        starts = [0]
    else:
        stride = max(1, (len(ids) - seq_len) // n_windows)
        starts = [i * stride for i in range(n_windows)]
        starts = [s for s in starts if s + seq_len <= len(ids)]
        if not starts:
            starts = [0]

    captured = {}
    def hook_fn(module, input, output):
        captured['hidden'] = output.detach()
    handle = model.norm.register_forward_hook(hook_fn)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        for s in starts:
            chunk = ids[s:s + seq_len]
            t = torch.tensor([chunk], dtype=torch.long, device=device)
            model(t)
            all_hidden.append(captured['hidden'][0].float())  # (L, d)

    handle.remove()

    # Pool across all windows
    cat = torch.cat(all_hidden, dim=0)  # (total_L, d)
    mean_emb = cat.mean(dim=0).cpu().numpy()
    std_emb = cat.std(dim=0).cpu().numpy()
    max_emb = cat.max(dim=0).values.cpu().numpy()
    return np.concatenate([mean_emb, std_emb, max_emb])


def balanced_probe(X_human, X_nonhuman, n_repeats=20, probe_type="logreg"):
    """Run balanced evaluation: undersample majority to match minority, repeat n times."""
    n_human = len(X_human)
    if n_human < 5:
        return None

    all_accs, all_aucs, all_f1s, all_mccs = [], [], [], []
    rng = np.random.RandomState(SEED)

    for rep in range(n_repeats):
        # Undersample non-human to match human count
        idx = rng.choice(len(X_nonhuman), size=n_human, replace=False)
        X_neg = X_nonhuman[idx]
        X = np.vstack([X_human, X_neg])
        y = np.array([1]*n_human + [0]*n_human)

        # Shuffle
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        # 5-fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        fold_accs, fold_aucs, fold_f1s, fold_mccs = [], [], [], []

        for tr_idx, te_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])

            if probe_type == "logreg":
                clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced',
                                         random_state=SEED)
            elif probe_type == "svm":
                clf = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True,
                          random_state=SEED)
            elif probe_type == "mlp":
                clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000,
                                     random_state=SEED, early_stopping=True)
            else:
                raise ValueError(f"Unknown probe: {probe_type}")

            clf.fit(X_tr, y[tr_idx])
            pred = clf.predict(X_te)
            prob = clf.predict_proba(X_te)[:, 1]
            fold_accs.append(accuracy_score(y[te_idx], pred))
            fold_f1s.append(f1_score(y[te_idx], pred))
            fold_mccs.append(matthews_corrcoef(y[te_idx], pred))
            try:
                fold_aucs.append(roc_auc_score(y[te_idx], prob))
            except ValueError:
                pass

        all_accs.append(np.mean(fold_accs))
        all_f1s.append(np.mean(fold_f1s))
        all_mccs.append(np.mean(fold_mccs))
        if fold_aucs:
            all_aucs.append(np.mean(fold_aucs))

    return {
        "accuracy": float(np.mean(all_accs)),
        "accuracy_std": float(np.std(all_accs)),
        "auroc": float(np.mean(all_aucs)) if all_aucs else float('nan'),
        "auroc_std": float(np.std(all_aucs)) if all_aucs else float('nan'),
        "f1": float(np.mean(all_f1s)),
        "f1_std": float(np.std(all_f1s)),
        "mcc": float(np.mean(all_mccs)),
        "mcc_std": float(np.std(all_mccs)),
    }


def main():
    set_seed(SEED)
    amp_dtype, _ = get_amp_settings()
    os.makedirs(OUT_DIR, exist_ok=True)

    records = parse_fasta_with_headers("data/attack.fasta", min_len=512)
    human_idx = [i for i, (h, _) in enumerate(records) if is_human_virus(h)]
    nonhuman_idx = [i for i, (h, _) in enumerate(records) if not is_human_virus(h)]
    print(f"Loaded {len(records)} virus sequences: {len(human_idx)} human, {len(nonhuman_idx)} non-human\n",
          flush=True)

    out_path = os.path.join(OUT_DIR, "tropism_results.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    for ckpt_name, ckpt_path in CHECKPOINTS:
        print(f"\n{'='*70}", flush=True)
        print(f"Checkpoint: {ckpt_name}", flush=True)
        print(f"{'='*70}", flush=True)

        model, tokenizer = load_evo_model("evo-1-8k-base", DEVICE)
        maybe_load_locked_checkpoint(model, ckpt_path)
        model.eval()

        with specdef_fused_eval(model):
            # --- Extract embeddings (single and multi-window) ---
            print("  Extracting embeddings...", flush=True)
            embs_single = []
            embs_multi = []
            for i, (header, seq) in enumerate(records):
                e1 = extract_embedding(model, tokenizer, seq, DEVICE, SEQ_LEN, amp_dtype)
                e2 = extract_multi_window_embedding(model, tokenizer, seq, DEVICE,
                                                     SEQ_LEN, amp_dtype, n_windows=3)
                embs_single.append(e1)
                embs_multi.append(e2)
                if (i + 1) % 100 == 0:
                    print(f"    [{i+1}/{len(records)}]", flush=True)

        # Build X arrays
        valid_single = [i for i in range(len(records)) if embs_single[i] is not None]
        valid_multi = [i for i in range(len(records)) if embs_multi[i] is not None]

        for emb_name, embs, valid_idx in [
            ("single_window", embs_single, valid_single),
            ("multi_window", embs_multi, valid_multi),
        ]:
            X_all = np.array([embs[i] for i in valid_idx])
            labels = np.array([1 if i in set(human_idx) else 0 for i in valid_idx])
            X_human = X_all[labels == 1]
            X_nonhuman = X_all[labels == 0]

            print(f"\n  [{emb_name}] {X_human.shape[0]} human, {X_nonhuman.shape[0]} non-human  dim={X_all.shape[1]}",
                  flush=True)

            for probe_type in ["logreg", "svm", "mlp"]:
                result = balanced_probe(X_human, X_nonhuman, n_repeats=20, probe_type=probe_type)
                if result is None:
                    continue
                record = {
                    "checkpoint": ckpt_name,
                    "embedding": emb_name,
                    "probe": probe_type,
                    **result,
                }
                print(f"    {probe_type:8s}: Acc={result['accuracy']:.3f}±{result['accuracy_std']:.3f}  "
                      f"AUROC={result['auroc']:.3f}±{result['auroc_std']:.3f}  "
                      f"F1={result['f1']:.3f}±{result['f1_std']:.3f}  "
                      f"MCC={result['mcc']:.3f}±{result['mcc_std']:.3f}", flush=True)

                with open(out_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

        del model
        torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY (best per checkpoint — multi_window + best probe)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Checkpoint':25s}  {'Probe':8s}  {'AUROC':>8s}  {'F1':>8s}  {'MCC':>8s}", flush=True)
    print("-" * 60, flush=True)

    with open(out_path) as f:
        lines = [json.loads(l) for l in f]

    for ckpt in [c[0] for c in CHECKPOINTS]:
        subset = [l for l in lines if l['checkpoint'] == ckpt and l['embedding'] == 'multi_window']
        if not subset:
            continue
        best = max(subset, key=lambda x: x.get('auroc', 0))
        print(f"{ckpt:25s}  {best['probe']:8s}  {best['auroc']:>8.3f}  {best['f1']:>8.3f}  {best['mcc']:>8.3f}",
              flush=True)

    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
