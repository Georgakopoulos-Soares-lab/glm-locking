"""Retain capability evaluation — tests whether locked/attacked models
preserve general genomic usefulness on non-attack data.

Two tasks:
  1. BACTERIA GENUS PROBE — Does the model still encode bacterial taxonomy?
     Extract embeddings from retain.fasta; train multi-class probe to predict genus.
     If locked model destroys retain representations, it's no longer useful.

  2. BACTERIA PPL — Simple perplexity on retain.fasta (50-seq sample).
     Already measured but re-run here for same-script consistency.

  3. GC-CONTENT REGRESSION — Does model log-prob correlate with GC content
     across the retain sequences? A general genomics model should track this.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n evo --no-capture-output \
        python -u scripts/eval_retain_capability.py
"""

import os, sys, json, math, random
from collections import Counter
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    set_seed, get_amp_settings, clean_dna,
    load_evo_model, maybe_load_locked_checkpoint,
    causal_lm_loss, specdef_fused_eval,
)

DEVICE  = "cuda:0"
SEQ_LEN = 1024
SEED    = 42
OUT_DIR = "results/eval_retain_capability"

CHECKPOINTS = [
    ("pretrained",       None),
    ("unlocked_lr1e5",   "results/ft_paper_lr1e5_unlocked/model_best.pt"),
    ("locked_lr1e5",     "results/ft_paper_lr1e5_locked/model_best.pt"),
    ("locked_lr3e5",     "results/ft_paper_lr3e5_locked/model_best.pt"),
]

# Top genera with enough samples for stratified CV (min 10 sequences)
MIN_GENUS_COUNT = 10


def parse_fasta(path, min_len=512):
    records = []
    header, chunks = None, []
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


def extract_genus(header):
    """Extract genus from NCBI header like 'NZ_... Escherichia coli strain ...'"""
    # Header format: "ACCESSION Species genus strain ..."
    parts = header.split()
    # Skip accession (first token), return first word of species name
    if len(parts) >= 2:
        return parts[1]
    return "Unknown"


def gc_content(seq):
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    total = sum(seq.count(b) for b in "ACGT")
    return gc / total if total > 0 else float("nan")


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
        captured["hidden"] = output.detach()
    handle = model.norm.register_forward_hook(hook_fn)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        model(t)
    handle.remove()
    return captured["hidden"][0].float().mean(dim=0).cpu().numpy()


def compute_ppl(model, tokenizer, seq, device, seq_len, amp_dtype):
    ids = list(tokenizer.tokenize(seq[:seq_len]))
    if len(ids) < 2:
        return float("nan")
    t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits, _ = model(t)
        loss = causal_lm_loss(logits, t)
    return math.exp(loss.item())


def task_genus_probe(X, genera, ckpt_name, n_repeats=10):
    """Multi-class genus classification with balanced evaluation."""
    le = LabelEncoder()
    y = le.fit_transform(genera)
    classes, counts = np.unique(y, return_counts=True)

    # Keep only genera with at least MIN_GENUS_COUNT sequences
    valid_classes = classes[counts >= MIN_GENUS_COUNT]
    mask = np.isin(y, valid_classes)
    X_f, y_f = X[mask], y[mask]

    # Re-encode
    le2 = LabelEncoder()
    y_f = le2.fit_transform(y_f)
    n_classes = len(np.unique(y_f))

    print(f"    Genus probe: {len(X_f)} sequences, {n_classes} genera (≥{MIN_GENUS_COUNT} seqs each)",
          flush=True)
    if n_classes < 2:
        return None

    all_accs, all_baccs, all_f1s = [], [], []
    rng = np.random.RandomState(SEED)

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        fold_accs, fold_baccs, fold_f1s = [], [], []
        for tr_idx, te_idx in skf.split(X_f, y_f):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_f[tr_idx])
            X_te = scaler.transform(X_f[te_idx])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                     random_state=SEED, multi_class="multinomial",
                                     solver="lbfgs")
            clf.fit(X_tr, y_f[tr_idx])
            pred = clf.predict(X_te)
            fold_accs.append(accuracy_score(y_f[te_idx], pred))
            fold_baccs.append(balanced_accuracy_score(y_f[te_idx], pred))
            fold_f1s.append(f1_score(y_f[te_idx], pred, average="macro"))
        all_accs.append(np.mean(fold_accs))
        all_baccs.append(np.mean(fold_baccs))
        all_f1s.append(np.mean(fold_f1s))

    return {
        "n_sequences": int(len(X_f)),
        "n_classes": int(n_classes),
        "accuracy": float(np.mean(all_accs)),
        "accuracy_std": float(np.std(all_accs)),
        "balanced_accuracy": float(np.mean(all_baccs)),
        "balanced_accuracy_std": float(np.std(all_baccs)),
        "macro_f1": float(np.mean(all_f1s)),
        "macro_f1_std": float(np.std(all_f1s)),
    }


def main():
    set_seed(SEED)
    amp_dtype, _ = get_amp_settings()
    os.makedirs(OUT_DIR, exist_ok=True)

    records = parse_fasta("data/retain.fasta", min_len=512)
    genera = [extract_genus(h) for h, _ in records]
    genus_counts = Counter(genera)
    print(f"Loaded {len(records)} bacterial sequences, {len(genus_counts)} genera", flush=True)
    print(f"Top 10: {genus_counts.most_common(10)}\n", flush=True)

    # Sample 300 sequences stratified for PPL (manageable runtime)
    rng = random.Random(SEED)
    ppl_sample = rng.sample(records, min(300, len(records)))

    out_path = os.path.join(OUT_DIR, "retain_results.jsonl")
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
            # ── Task 1: Bacteria PPL ──────────────────────────────────────────
            print("\n  [Task 1] Bacteria PPL (300 sequences)", flush=True)
            ppls = []
            for i, (header, seq) in enumerate(ppl_sample):
                ppl = compute_ppl(model, tokenizer, seq, DEVICE, SEQ_LEN, amp_dtype)
                ppls.append(ppl)
                if (i + 1) % 50 == 0:
                    valid = [p for p in ppls if not math.isnan(p)]
                    print(f"    [{i+1}/300]  running mean PPL={sum(valid)/len(valid):.3f}",
                          flush=True)

            valid_ppls = [p for p in ppls if not math.isnan(p)]
            ppl_result = {
                "mean_ppl": float(np.mean(valid_ppls)),
                "std_ppl": float(np.std(valid_ppls)),
                "n": len(valid_ppls),
            }
            print(f"    -> mean PPL={ppl_result['mean_ppl']:.4f} ± {ppl_result['std_ppl']:.4f}",
                  flush=True)

            # ── Task 2: GC content correlation ───────────────────────────────
            print("\n  [Task 2] GC-content vs log-prob correlation", flush=True)
            gc_vals, lp_vals = [], []
            for header, seq in ppl_sample:
                ids = list(tokenizer.tokenize(seq[:SEQ_LEN]))
                if len(ids) < 2:
                    continue
                t = torch.tensor([ids], dtype=torch.long, device=DEVICE)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = model(t)
                    loss = causal_lm_loss(logits, t)
                gc_vals.append(gc_content(seq))
                lp_vals.append(-loss.item())  # log-prob (higher = more likely)

            rho, p_val = spearmanr(gc_vals, lp_vals)
            gc_result = {"rho": float(rho), "p": float(p_val), "n": len(gc_vals)}
            print(f"    -> ρ={rho:.4f}  p={p_val:.3e}  (n={len(gc_vals)})", flush=True)

            # ── Task 3: Genus embedding probe (all sequences) ─────────────────
            print("\n  [Task 3] Bacteria genus classification probe", flush=True)
            print(f"    Extracting embeddings for {len(records)} sequences...", flush=True)
            embeddings = []
            for i, (header, seq) in enumerate(records):
                emb = extract_embedding(model, tokenizer, seq, DEVICE, SEQ_LEN, amp_dtype)
                embeddings.append(emb)
                if (i + 1) % 200 == 0:
                    print(f"    [{i+1}/{len(records)}]", flush=True)

            valid_idx = [i for i, e in enumerate(embeddings) if e is not None]
            X = np.array([embeddings[i] for i in valid_idx])
            g = [genera[i] for i in valid_idx]

            probe_result = task_genus_probe(X, g, ckpt_name, n_repeats=10)
            if probe_result:
                print(f"    -> BalAcc={probe_result['balanced_accuracy']:.3f}±{probe_result['balanced_accuracy_std']:.3f}  "
                      f"MacroF1={probe_result['macro_f1']:.3f}±{probe_result['macro_f1_std']:.3f}  "
                      f"Acc={probe_result['accuracy']:.3f}±{probe_result['accuracy_std']:.3f}",
                      flush=True)

        record = {
            "checkpoint": ckpt_name,
            "bacteria_ppl": ppl_result,
            "gc_correlation": gc_result,
            "genus_probe": probe_result,
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        del model
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Checkpoint':25s}  {'BactPPL':>8s}  {'GC ρ':>8s}  {'BalAcc':>8s}  {'MacroF1':>9s}", flush=True)
    print("-" * 65, flush=True)
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            ppl = r["bacteria_ppl"]["mean_ppl"]
            rho = r["gc_correlation"]["rho"]
            gp = r["genus_probe"] or {}
            ba = gp.get("balanced_accuracy", float("nan"))
            mf = gp.get("macro_f1", float("nan"))
            print(f"{r['checkpoint']:25s}  {ppl:>8.4f}  {rho:>+8.4f}  {ba:>8.3f}  {mf:>9.3f}", flush=True)

    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
