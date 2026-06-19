"""ViroBench host-prediction probe with a pre-committed k-mer composition baseline.

Independent, third-party cross-check of whether ANY Evo checkpoint carries
host-prediction signal *beyond sequence composition* on the ViroBench corpus,
and whether weight-locking (M, α=3×10⁵) degrades that signal relative to the
unlocked fine-tune.

Why this design (read me):
  * ViroBench's `genus` directory is NOT phylogeny-disjoint (train/test share
    ~97% of families, 83% of genera) — k-mer composition transfers across it.
    Its `times` directory is a clean TEMPORAL split (train ≤2017-10, test
    ≥2020-02): the biosecurity-relevant question of predicting the host of
    newly-emerged viruses. We use `times`.
  * Because families/genera still overlap even in the temporal split, ViroBench
    does NOT automatically defeat the composition confound. Therefore — exactly
    as in scripts/hvue_rigorous_compare.py — we ALWAYS report a matched k-mer
    baseline on the IDENTICAL (truncated) sequences and the residual
    (model − k-mer) with a paired bootstrap 95% CI. A model result only counts
    if its residual over k-mer is positive and significant.

Protocol (matched, leakage-free):
  1. SAME split        — model and k-mer trained on `train`, reported on `test`.
  2. SAME sequences     — k-mer features computed on the EXACT truncated rows the
                          model embeds (one shared subsample, fixed seed).
  3. SAME label set     — host_label ∈ {A,B,C,D1,D2,D3,E,F}, multiclass.
  4. NO test leakage    — probe C chosen by 5-fold CV on `train`, reported on `test`.
  5. FUSED locked eval  — locked checkpoints run through specdef_fused_eval so the
                          attack's weight changes are evaluated on a clean bf16 path
                          (fixes the raw-bf16 handicap in hvue_extract_one_ckpt.py).
  6. Residual capability— model macro-AUROC, k-mer macro-AUROC, residual + bootstrap CI.

Stage 1 (GPU): extract mean-pooled final-norm embeddings per checkpoint (cached).
Stage 2 (CPU): fit multiclass probes, k-mer baseline, residual bootstrap, write CSV.

Usage:
  CUDA_VISIBLE_DEVICES=7 python scripts/virobench_host_probe.py \
      --ckpts pretrained unlocked_ft M_a300k locked_no_ft \
      --out experiments/exp3_virobench/host_times_probe.csv
"""
from __future__ import annotations
import os, sys, csv, json, time, argparse, itertools, contextlib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
import torch.nn as nn
from src.utils import (load_evo_model, maybe_load_locked_checkpoint,
                       get_amp_settings)

# ── Config ────────────────────────────────────────────────────────────────────
VIROBENCH_SNAP = (
    "/data/huggingface_cache/hub/datasets--YDXX--ViroBench/"
    "snapshots/77461edda1bc4f3c0ea96008384a678fefe04a14"
)
SPLIT_DIR = "Classification/ALL/host/times"   # temporal split, host_label target
HOST_CLASSES = ["A", "B", "C", "D1", "D2", "D3", "E", "F"]

N_TRAIN = 4000        # stratified subsample for probe training / embedding
N_TEST = 3000         # stratified subsample for reporting
MAX_LEN = 4096        # bases (= Evo tokens) kept per sequence
BATCH = 2
KMER_K = 4
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
N_BOOT = 5000
SEED = 0
NUCLEOTIDES = ['A', 'C', 'G', 'T']

EMB_DIR = "experiments/exp3_virobench/host_times_embeddings"

# checkpoint registry: name -> path (None = pretrained, "" handled as None)
CKPTS = {
    "pretrained":  None,
    "locked_no_ft": "results/lock_alpha300k/model_specdef.pt",
    "unlocked_ft": "results/ft_unlocked_full910_25k_unlocked/model_finetuned.pt",
    "M_a300k":     "results/ft_locked_a300k_lr1e5_25k_locked/model_finetuned.pt",
}


# ── Data: build ONE shared, deterministic subsample (seq + label) ─────────────

def _load_split(split: str) -> pd.DataFrame:
    """Join host_label meta (csv) with sequences (jsonl) on taxid for one split."""
    meta = pd.read_csv(f"{VIROBENCH_SNAP}/{SPLIT_DIR}/{split}.csv")
    rows = [json.loads(l) for l in open(f"{VIROBENCH_SNAP}/{SPLIT_DIR}/{split}_sequences.jsonl")]
    seqdf = pd.DataFrame({
        "taxid": [r["taxid"] for r in rows],
        "sequence": [r["sequences"][0] for r in rows],
    })
    meta = meta[["taxid", "host_label"]].dropna()
    df = meta.merge(seqdf, on="taxid", how="inner")
    df = df[df["host_label"].isin(HOST_CLASSES)].copy()
    return df


def _stratified_subsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Stratified subsample of n rows by host_label (proportional, deterministic)."""
    if len(df) <= n:
        return df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    frac = n / len(df)
    sub = df.groupby("host_label", group_keys=False).apply(
        lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=SEED)
    )
    sub = sub.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return sub


def build_shared_data():
    """Return (train_df, test_df) with truncated sequences and integer labels.

    Cached to disk so every checkpoint and the k-mer baseline see IDENTICAL rows.
    """
    os.makedirs(EMB_DIR, exist_ok=True)
    cache = f"{EMB_DIR}/shared_data.parquet"
    if os.path.exists(cache):
        all_df = pd.read_parquet(cache)
        tr = all_df[all_df["_split"] == "train"].reset_index(drop=True)
        te = all_df[all_df["_split"] == "test"].reset_index(drop=True)
        return tr, te
    tr = _stratified_subsample(_load_split("train"), N_TRAIN)
    te = _stratified_subsample(_load_split("test"), N_TEST)
    label2id = {c: i for i, c in enumerate(HOST_CLASSES)}
    for d, name in [(tr, "train"), (te, "test")]:
        d["sequence"] = d["sequence"].str.slice(0, MAX_LEN)
        d["y"] = d["host_label"].map(label2id).astype(int)
        d["_split"] = name
    pd.concat([tr, te], ignore_index=True).to_parquet(cache)
    print(f"[data] train={len(tr)} test={len(te)}  -> {cache}")
    for name, d in [("train", tr), ("test", te)]:
        dist = d["host_label"].value_counts().reindex(HOST_CLASSES).fillna(0).astype(int).to_dict()
        print(f"  {name} host_label dist: {dist}")
    return tr.reset_index(drop=True), te.reset_index(drop=True)


# ── Stage 1: GPU embedding extraction (cached per checkpoint) ─────────────────

def find_norm(model):
    for name, mod in model.named_modules():
        if name == 'norm' or name.endswith('.norm'):
            return mod
    raise RuntimeError("no final norm module found")


@contextlib.contextmanager
def fused_eval_f64(model):
    """Fuse SpecDef wrappers using a FLOAT64 product C@W̃, then bf16 single matmul.

    Why not src.utils.specdef_fused_eval: that helper forms C@W̃ in float32. For
    checkpoints whose compensation matrix C was stored in bfloat16 (e.g. M_a300k,
    where cast_comp_bf16=True), the f32 product suffers catastrophic cancellation
    (W̃ has σ_max≈4.85e6, fro-norm≈6.8e6; the result should be a moderate-norm W),
    losing ~4-5 significant digits and corrupting the eval (training-logged
    val_ppl≈7 vs true≈3.84). Computing the SAME product C@W̃ in float64 handles the
    cancellation accurately and reconstructs the model's true effective weight
    (training used this exact C, so C@W̃ in f64 IS the trained function), then we
    cast to bf16 so every checkpoint runs the IDENTICAL single-bf16-matmul path
    as the pretrained/unlocked models. No-op if no SpecDef layers are present.
    """
    from scripts.lock_specdef import _find_specdef_layers, _set_nested
    found = _find_specdef_layers(model)
    if not found:
        yield
        return
    saved = {}
    with torch.no_grad():
        for idx, pattern, wrapper in found:
            saved[(idx, pattern)] = wrapper
            fused_w = (wrapper.comp.weight.data.double()
                       @ wrapper.linear.weight.data.double())  # C@W̃ in f64
            d_out, d_in = wrapper.linear.weight.shape
            has_bias = wrapper.bias is not None
            fused = nn.Linear(d_in, d_out, bias=has_bias,
                              dtype=torch.bfloat16, device=fused_w.device)
            fused.weight.data = fused_w.bfloat16()
            if has_bias:
                fused.bias.data = wrapper.bias.data.bfloat16()
            _set_nested(model.blocks[idx], pattern, fused)
    try:
        yield
    finally:
        for (idx, pattern), wrapper in saved.items():
            _set_nested(model.blocks[idx], pattern, wrapper)


def extract_embeddings(ckpt_name, ckpt_path, tr, te, device):
    out = f"{EMB_DIR}/{ckpt_name}.npz"
    if os.path.exists(out):
        print(f"[emb] skip exists: {out}")
        return out

    amp_dtype, _ = get_amp_settings()
    model, tok = load_evo_model('evo-1-8k-base', device)
    is_locked = bool(ckpt_path) and ("locked" in ckpt_name or "M_" in ckpt_name)
    maybe_load_locked_checkpoint(model, ckpt_path if ckpt_path else None)
    model.eval()

    @torch.no_grad()
    def hidden_for(seqs, hook_target):
        captured = {}
        def hook(_m, _i, o):
            captured['h'] = o.detach()
        h = hook_target.register_forward_hook(hook)
        ids_list = [tok.tokenize(s) for s in seqs]
        L = max(len(x) for x in ids_list)
        ids = torch.zeros((len(ids_list), L), dtype=torch.long)
        for j, x in enumerate(ids_list):
            ids[j, :len(x)] = torch.tensor(x)
        ids = ids.to(device)
        try:
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                _ = model(ids)
        finally:
            h.remove()
        return captured['h'].float().mean(dim=1).cpu().numpy()

    # Route locked checkpoints through the f64-fused bf16 path (matched & accurate
    # for both f32-C and bf16-C checkpoints); no-op for unlocked models.
    ctx = fused_eval_f64(model) if is_locked else contextlib.nullcontext()
    feats = {}
    with ctx:
        hook_target = find_norm(model)
        for split_name, d in [("train", tr), ("test", te)]:
            X = []
            t0 = time.time()
            seqs = d["sequence"].tolist()
            for i in range(0, len(seqs), BATCH):
                batch = seqs[i:i + BATCH]
                try:
                    X.append(hidden_for(batch, hook_target))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    for s in batch:
                        X.append(hidden_for([s], hook_target))
                if (i // BATCH) % 100 == 0 and i > 0:
                    rate = (i + BATCH) / (time.time() - t0)
                    print(f"   [{ckpt_name}/{split_name}] {i + BATCH}/{len(seqs)} ({rate:.1f} seq/s)")
            feats[split_name] = np.concatenate(X, axis=0)
            print(f"   [{ckpt_name}/{split_name}] X={feats[split_name].shape} in {time.time()-t0:.1f}s")
    np.savez_compressed(out, X_train=feats["train"], X_test=feats["test"],
                        y_train=tr["y"].values, y_test=te["y"].values,
                        fused=is_locked)
    del model
    torch.cuda.empty_cache()
    print(f"[emb] wrote {out}")
    return out


# ── k-mer composition features ────────────────────────────────────────────────

def kmer_frequencies(seqs, k: int) -> np.ndarray:
    kmers = [''.join(p) for p in itertools.product(NUCLEOTIDES, repeat=k)]
    idx = {km: i for i, km in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s.upper()
        for j in range(len(s) - k + 1):
            km = s[j:j + k]
            if km in idx:
                X[i, idx[km]] += 1
        tot = X[i].sum()
        if tot > 0:
            X[i] /= tot
    return X


# ── Stage 2: multiclass probe (C by CV on train, report on test) ──────────────

def macro_auroc(y_true, proba, classes):
    """One-vs-rest macro AUROC, averaged only over classes present in y_true."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for ci, _c in enumerate(classes):
        yt = (y_true == ci).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, proba[:, ci]))
    return float(np.mean(aucs)) if aucs else float('nan')


def fit_probe(X_tr, y_tr, X_te):
    """Multinomial LR; C by 5-fold CV (macro-ovr AUROC) on train. Returns test proba."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_C, best = None, -np.inf
    for C in C_GRID:
        clf = LogisticRegression(max_iter=3000, C=C, random_state=SEED)
        s = cross_val_score(clf, X_tr_s, y_tr, cv=cv,
                            scoring='roc_auc_ovr', n_jobs=-1).mean()
        if s > best:
            best, best_C = s, C
    clf = LogisticRegression(max_iter=3000, C=best_C, random_state=SEED)
    clf.fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s), best_C, clf.classes_


def align_proba(proba, clf_classes, n_classes):
    """Map a classifier's proba columns onto the full HOST_CLASSES index space."""
    full = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
    for j, c in enumerate(clf_classes):
        full[:, int(c)] = proba[:, j]
    return full


def boot_residual_ci(y_te, model_proba, kmer_proba, classes, n_boot=N_BOOT):
    """Paired bootstrap over test rows: residual = model macroAUROC − kmer macroAUROC."""
    rng = np.random.default_rng(SEED)
    n = len(y_te)
    res = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = macro_auroc(y_te[idx], model_proba[idx], classes)
        k = macro_auroc(y_te[idx], kmer_proba[idx], classes)
        if not (np.isnan(m) or np.isnan(k)):
            res.append(m - k)
    res = np.array(res)
    return float(np.mean(res)), float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', default=list(CKPTS.keys()))
    ap.add_argument('--kmer_k', type=int, default=KMER_K)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='experiments/exp3_virobench/host_times_probe.csv')
    ap.add_argument('--extract_only', action='store_true')
    a = ap.parse_args()

    tr, te = build_shared_data()

    # Stage 1: GPU extraction (cached)
    for name in a.ckpts:
        if name not in CKPTS:
            print(f"[warn] unknown ckpt '{name}', skipping"); continue
        extract_embeddings(name, CKPTS[name], tr, te, a.device)
    if a.extract_only:
        print("extract_only: done"); return

    # k-mer baseline on the IDENTICAL truncated rows
    print(f"\n[kmer] computing k={a.kmer_k} features on shared rows...")
    Xk_tr = kmer_frequencies(tr["sequence"].tolist(), a.kmer_k)
    Xk_te = kmer_frequencies(te["sequence"].tolist(), a.kmer_k)
    y_tr, y_te = tr["y"].values, te["y"].values
    nC = len(HOST_CLASSES)
    kp, kC, kcls = fit_probe(Xk_tr, y_tr, Xk_te)
    kmer_proba = align_proba(kp, kcls, nC)
    kmer_auc = macro_auroc(y_te, kmer_proba, range(nC))
    print(f"[kmer] macro-AUROC(test) = {kmer_auc:.4f}  (C={kC})")

    # Stage 2: per-checkpoint probe + residual
    rows = []
    for name in a.ckpts:
        npz = f"{EMB_DIR}/{name}.npz"
        if not os.path.exists(npz):
            print(f"[warn] no embeddings for {name}; skipping"); continue
        d = np.load(npz, allow_pickle=True)
        mp, mC, mcls = fit_probe(d["X_train"], d["y_train"], d["X_test"])
        model_proba = align_proba(mp, mcls, nC)
        m_auc = macro_auroc(d["y_test"], model_proba, range(nC))
        rmean, rlo, rhi = boot_residual_ci(d["y_test"], model_proba, kmer_proba, range(nC))
        sig = "yes" if rlo > 0 else ("neg" if rhi < 0 else "ns")
        rows.append(dict(ckpt=name, fused=bool(d["fused"]), n_test=len(d["y_test"]),
                         model_auroc=round(m_auc, 4), kmer_auroc=round(kmer_auc, 4),
                         residual=round(rmean, 4), ci_lo=round(rlo, 4), ci_hi=round(rhi, 4),
                         beats_kmer=sig, model_C=mC, kmer_C=kC, kmer_k=a.kmer_k))
        print(f"[probe] {name:14s} model={m_auc:.4f} kmer={kmer_auc:.4f} "
              f"residual={rmean:+.4f} [{rlo:+.4f},{rhi:+.4f}] beats_kmer={sig}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[done] wrote {a.out}")


if __name__ == '__main__':
    main()
