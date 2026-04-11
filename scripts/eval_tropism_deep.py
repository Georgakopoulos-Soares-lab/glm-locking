"""Deep tropism representation analysis — disambiguates collapse vs geometry shift.

Tests the key question: Are locked model representations *truly random* (collapsed)
or geometrically reorganized (linear probe fails but structure still exists)?

Methods:
  1. Fisher Discriminant Ratio (FDR) — between-class vs within-class variance
  2. kNN probe (k=1, k=5, k=15) — non-linear, finds local structure
  3. Silhouette score — cluster quality independent of probe type
  4. Deep PyTorch MLP (512->256->128, Adam, dropout) — higher capacity probe
  5. PCA explained variance — is locked embedding space lower-dimensional?

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -n evo --no-capture-output \
        python -u scripts/eval_tropism_deep.py > results/eval_tropism_deep.log 2>&1
"""

import os, sys, json, math, random, warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    set_seed, get_amp_settings, clean_dna,
    load_evo_model, maybe_load_locked_checkpoint,
    specdef_fused_eval,
)

DEVICE   = "cuda:0"
SEQ_LEN  = 1024
SEED     = 42
N_REPEATS = 10   # repeats for balanced undersampling
OUT_DIR  = "results/bio_tropism_deep"

CHECKPOINTS = [
    ("pretrained",       None),
    ("unlocked_lr1e5",   "results/ft_paper_lr1e5_unlocked/model_best.pt"),
    ("locked_lr1e5",     "results/ft_paper_lr1e5_locked/model_best.pt"),
    ("locked_lr3e5",     "results/ft_paper_lr3e5_locked/model_best.pt"),
    ("locked_lr1e4",     "results/ft_paper_lr1e4_locked/model_best.pt"),
]


# ---------------------------------------------------------------------------
# Data loading (identical to eval_tropism_balanced.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_multi_window_embedding(model, tokenizer, seq, device, seq_len, amp_dtype, n_windows=3):
    ids = list(tokenizer.tokenize(seq[:seq_len * 8]))
    if len(ids) < 10:
        return None
    all_hidden = []
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
            all_hidden.append(captured['hidden'][0].float())

    handle.remove()
    cat = torch.cat(all_hidden, dim=0)
    mean_emb = cat.mean(dim=0).cpu().numpy()
    std_emb  = cat.std(dim=0).cpu().numpy()
    max_emb  = cat.max(dim=0).values.cpu().numpy()
    return np.concatenate([mean_emb, std_emb, max_emb])


# ---------------------------------------------------------------------------
# Analysis methods
# ---------------------------------------------------------------------------

def fisher_discriminant_ratio(X, y):
    """Compute trace(S_B) / trace(S_W) — class-separability metric."""
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    S_W = np.zeros(X.shape[1])  # within-class variance (diagonal)
    S_B = np.zeros(X.shape[1])  # between-class variance (diagonal)
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(axis=0)
        S_W += ((Xc - mc) ** 2).sum(axis=0)
        n_c = len(Xc)
        diff = mc - overall_mean
        S_B += n_c * (diff ** 2)
    # Use trace (sum of diagonal)
    tr_SW = S_W.sum() + 1e-12
    tr_SB = S_B.sum()
    return float(tr_SB / tr_SW)


def knn_probe(X_human, X_nonhuman, k_values=(1, 5, 15), n_repeats=10, rng_seed=42):
    """Balanced kNN evaluation."""
    rng = np.random.RandomState(rng_seed)
    n_human = len(X_human)
    results = {k: [] for k in k_values}

    for _ in range(n_repeats):
        idx = rng.choice(len(X_nonhuman), size=n_human, replace=False)
        X = np.vstack([X_human, X_nonhuman[idx]])
        y = np.array([1]*n_human + [0]*n_human)
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng_seed)
        fold_acc = {k: [] for k in k_values}
        for tr_idx, te_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])
            for k in k_values:
                clf = KNeighborsClassifier(n_neighbors=k, metric='cosine')
                clf.fit(X_tr, y[tr_idx])
                pred = clf.predict(X_te)
                fold_acc[k].append(accuracy_score(y[te_idx], pred))
        for k in k_values:
            results[k].append(np.mean(fold_acc[k]))

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}


class DeepMLP(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def deep_mlp_probe(X_human, X_nonhuman, n_repeats=10, rng_seed=42):
    """Deep PyTorch MLP probe with Adam, dropout, batch norm."""
    rng = np.random.RandomState(rng_seed)
    n_human = len(X_human)
    all_aucs, all_mccs = [], []
    mlp_device = "cpu"  # run on CPU to avoid CUDA OOM (embeddings already extracted)

    for rep in range(n_repeats):
        idx = rng.choice(len(X_nonhuman), size=n_human, replace=False)
        X = np.vstack([X_human, X_nonhuman[idx]])
        y = np.array([1]*n_human + [0]*n_human)
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng_seed + rep)
        fold_aucs, fold_mccs = [], []

        for tr_idx, te_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx]).astype(np.float32)
            X_te = scaler.transform(X[te_idx]).astype(np.float32)

            model = DeepMLP(X_tr.shape[1]).to(mlp_device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            criterion = nn.CrossEntropyLoss()

            X_tr_t = torch.tensor(X_tr).to(mlp_device)
            y_tr_t = torch.tensor(y[tr_idx]).long().to(mlp_device)

            model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                logits = model(X_tr_t)
                loss = criterion(logits, y_tr_t)
                loss.backward()
                optimizer.step()
                scheduler.step()

            model.eval()
            with torch.no_grad():
                X_te_t = torch.tensor(X_te).to(mlp_device)
                logits_te = model(X_te_t)
                probs = torch.softmax(logits_te, dim=1)[:, 1].numpy()
                preds = logits_te.argmax(dim=1).numpy()

            try:
                fold_aucs.append(roc_auc_score(y[te_idx], probs))
            except ValueError:
                pass
            fold_mccs.append(matthews_corrcoef(y[te_idx], preds))

        if fold_aucs:
            all_aucs.append(np.mean(fold_aucs))
        all_mccs.append(np.mean(fold_mccs))

    return {
        "auroc": float(np.mean(all_aucs)) if all_aucs else float("nan"),
        "auroc_std": float(np.std(all_aucs)) if all_aucs else float("nan"),
        "mcc": float(np.mean(all_mccs)),
        "mcc_std": float(np.std(all_mccs)),
    }


def pca_analysis(X_human, X_nonhuman, n_components=50):
    """PCA: explained variance and effective dimensionality."""
    X_all = np.vstack([X_human, X_nonhuman])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    n_comp = min(n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cumvar, 0.90)) + 1  # dims for 90% var
    n_99 = int(np.searchsorted(cumvar, 0.99)) + 1  # dims for 99% var

    # Effective dimensionality (participation ratio)
    eva = pca.explained_variance_ratio_
    eff_dim = float((eva.sum()**2) / (eva**2).sum())

    return {
        "n_dims_for_90pct": int(n_90),
        "n_dims_for_99pct": int(n_99),
        "effective_dim": eff_dim,
        "top5_var": float(cumvar[4]) if n_comp >= 5 else float("nan"),
    }


def compute_silhouette(X_human, X_nonhuman, n_repeats=5, rng_seed=42):
    """Silhouette score on balanced subset."""
    rng = np.random.RandomState(rng_seed)
    n_human = len(X_human)
    scores = []
    scaler = StandardScaler()

    for _ in range(n_repeats):
        idx = rng.choice(len(X_nonhuman), size=n_human, replace=False)
        X = np.vstack([X_human, X_nonhuman[idx]])
        y = np.array([1]*n_human + [0]*n_human)
        X_sc = scaler.fit_transform(X)
        try:
            scores.append(silhouette_score(X_sc, y, metric='cosine', sample_size=None))
        except Exception:
            pass

    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    amp_dtype, _ = get_amp_settings()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "deep_results.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    records = parse_fasta_with_headers("data/attack.fasta", min_len=512)
    human_idx   = {i for i, (h, _) in enumerate(records) if is_human_virus(h)}
    nonhuman_idx = {i for i, (h, _) in enumerate(records) if not is_human_virus(h)}
    print(f"Loaded {len(records)} virus sequences: "
          f"{len(human_idx)} human, {len(nonhuman_idx)} non-human\n", flush=True)

    for ckpt_name, ckpt_path in CHECKPOINTS:
        print(f"\n{'='*70}", flush=True)
        print(f"Checkpoint: {ckpt_name}", flush=True)
        print(f"{'='*70}", flush=True)

        model, tokenizer = load_evo_model("evo-1-8k-base", DEVICE)
        maybe_load_locked_checkpoint(model, ckpt_path)
        model.eval()

        embeddings = []
        with specdef_fused_eval(model):
            print("  Extracting multi-window embeddings...", flush=True)
            for i, (header, seq) in enumerate(records):
                e = extract_multi_window_embedding(
                    model, tokenizer, seq, DEVICE, SEQ_LEN, amp_dtype, n_windows=3)
                embeddings.append(e)
                if (i + 1) % 100 == 0:
                    print(f"    [{i+1}/{len(records)}]", flush=True)

        del model
        torch.cuda.empty_cache()

        valid = [i for i in range(len(records)) if embeddings[i] is not None]
        X_all = np.array([embeddings[i] for i in valid])
        y_all = np.array([1 if i in human_idx else 0 for i in valid])
        X_human    = X_all[y_all == 1]
        X_nonhuman = X_all[y_all == 0]

        print(f"  Embedding dim: {X_all.shape[1]}, "
              f"human={X_human.shape[0]}, non-human={X_nonhuman.shape[0]}", flush=True)

        # 1. Fisher discriminant ratio
        print("  Computing Fisher discriminant ratio...", flush=True)
        fdr = fisher_discriminant_ratio(X_all, y_all)
        print(f"    FDR = {fdr:.4e}", flush=True)

        # 2. Silhouette score
        print("  Computing silhouette score...", flush=True)
        sil = compute_silhouette(X_human, X_nonhuman, n_repeats=5)
        print(f"    Silhouette = {sil:.4f}", flush=True)

        # 3. PCA analysis
        print("  Running PCA analysis...", flush=True)
        pca_res = pca_analysis(X_human, X_nonhuman, n_components=100)
        print(f"    n_dims(90%) = {pca_res['n_dims_for_90pct']},  "
              f"n_dims(99%) = {pca_res['n_dims_for_99pct']},  "
              f"eff_dim = {pca_res['effective_dim']:.1f},  "
              f"top5_var = {pca_res['top5_var']:.3f}", flush=True)

        # 4. kNN probe
        print("  Running kNN probe (k=1,5,15)...", flush=True)
        knn_res = knn_probe(X_human, X_nonhuman, k_values=(1, 5, 15),
                            n_repeats=N_REPEATS, rng_seed=SEED)
        for k, (mean_acc, std_acc) in knn_res.items():
            print(f"    kNN(k={k:2d}): acc = {mean_acc:.3f} ± {std_acc:.3f}", flush=True)

        # 5. Deep PyTorch MLP
        print("  Running deep MLP probe (512->256->128)...", flush=True)
        mlp_res = deep_mlp_probe(X_human, X_nonhuman,
                                  n_repeats=N_REPEATS, rng_seed=SEED)
        print(f"    Deep MLP: AUROC = {mlp_res['auroc']:.3f} ± {mlp_res['auroc_std']:.3f}  "
              f"MCC = {mlp_res['mcc']:.3f} ± {mlp_res['mcc_std']:.3f}", flush=True)

        record = {
            "checkpoint": ckpt_name,
            "n_human": int(X_human.shape[0]),
            "n_nonhuman": int(X_nonhuman.shape[0]),
            "embedding_dim": int(X_all.shape[1]),
            "fisher_discriminant_ratio": fdr,
            "silhouette": sil,
            "pca": pca_res,
            "knn": {f"k{k}": {"acc": m, "std": s} for k, (m, s) in knn_res.items()},
            "deep_mlp": mlp_res,
        }

        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    header_fmt = f"{'Checkpoint':25s}  {'FDR':>10s}  {'Sil':>7s}  {'EffDim':>7s}  {'kNN-5':>7s}  {'MLP AUROC':>10s}  {'MLP MCC':>8s}"
    print(header_fmt, flush=True)
    print("-" * 90, flush=True)

    with open(out_path) as f:
        rows = [json.loads(l) for l in f]

    for r in rows:
        knn5 = r["knn"]["k5"]["acc"]
        print(f"{r['checkpoint']:25s}  "
              f"{r['fisher_discriminant_ratio']:>10.4e}  "
              f"{r['silhouette']:>7.4f}  "
              f"{r['pca']['effective_dim']:>7.1f}  "
              f"{knn5:>7.3f}  "
              f"{r['deep_mlp']['auroc']:>10.3f}  "
              f"{r['deep_mlp']['mcc']:>8.3f}", flush=True)

    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
