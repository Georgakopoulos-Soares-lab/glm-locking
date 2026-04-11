"""Downstream biological evaluation of locked vs unlocked Evo models.

Three tasks that go beyond PPL to test real biological understanding:

1. HOST TROPISM PROBE — Extract embeddings, train linear probe to classify
   human-infecting vs non-human viruses. Tests if model encodes host-specificity.

2. CpG DINUCLEOTIDE SUPPRESSION — Measure if model's per-position log-probs
   correlate with observed CpG observed/expected ratio. Human-infecting viruses
   suppress CpG (innate immune evasion via ZAP/TLR9). A biologically-aware model
   should assign lower probability to CpG-rich regions in human viruses.

3. CODON ADAPTATION — For coding sequences, compare model likelihood at
   synonymous codon positions with the human Codon Adaptation Index (CAI).
   Human viruses optimize codons for human tRNA pools; a good model should
   assign higher probability to human-adapted codons.

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -n evo --no-capture-output \
        python -u scripts/eval_bio_downstream.py
"""

import os, sys, json, math, re, random
from collections import defaultdict
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    set_seed, get_amp_settings, load_sequences, clean_dna,
    load_evo_model, maybe_load_locked_checkpoint,
    causal_lm_loss, specdef_fused_eval,
)

DEVICE   = "cuda:0"
SEQ_LEN  = 1024
SEED     = 42
OUT_DIR  = "results/bio_downstream"

CHECKPOINTS = [
    ("pretrained",       None),
    ("unlocked_lr1e5",   "results/ft_paper_lr1e5_unlocked/model_best.pt"),
    ("locked_lr1e5",     "results/ft_paper_lr1e5_locked/model_best.pt"),
    ("locked_lr3e5",     "results/ft_paper_lr3e5_locked/model_best.pt"),
]

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def parse_fasta_with_headers(path: str, min_len: int = 512):
    """Return list of (header, sequence) tuples."""
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


def is_human_virus(header: str) -> bool:
    """Heuristic: is this a human-infecting virus based on FASTA header?"""
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
    # Explicit "Human" in header
    if "human" in h:
        return True
    for kw in human_keywords:
        if kw in h:
            return True
    return False


def extract_embedding(model, tokenizer, seq: str, device: str,
                      seq_len: int, amp_dtype) -> np.ndarray:
    """Extract mean-pooled embedding from the last hidden layer (before unembed).

    We hook into model.norm output (post-norm, pre-unembed).
    """
    ids = list(tokenizer.tokenize(seq[:seq_len * 2]))  # cap to ~2x window
    if len(ids) > seq_len:
        # Take middle window for most representative embedding
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

    hidden = captured['hidden'][0]  # (seq_len, d_model)
    # Mean pool
    emb = hidden.float().mean(dim=0).cpu().numpy()
    return emb


def get_per_position_logprobs(model, tokenizer, seq: str, device: str,
                              seq_len: int, amp_dtype) -> np.ndarray:
    """Get per-position log-probabilities for the actual tokens.

    Returns array of shape (N-1,) where N = min(len(ids), seq_len).
    logprobs[i] = log P(token_{i+1} | token_{0..i})
    """
    ids = list(tokenizer.tokenize(seq[:seq_len * 2]))
    if len(ids) > seq_len:
        ids = ids[:seq_len]
    if len(ids) < 10:
        return None
    t = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits, _ = model(t)

    # logprobs for actual next token
    log_probs = F.log_softmax(logits[0, :-1].float(), dim=-1)  # (N-1, vocab)
    targets = t[0, 1:]  # (N-1,)
    per_pos = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N-1,)
    return per_pos.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# TASK 1: Host Tropism Linear Probe
# ═══════════════════════════════════════════════════════════════════════════

def task_host_tropism(model, tokenizer, records, device, amp_dtype, ckpt_name):
    """Train linear probe on embeddings to classify human vs non-human viruses."""
    print("\n  [TASK 1] Host Tropism Probe", flush=True)

    embeddings = []
    labels = []
    for i, (header, seq) in enumerate(records):
        emb = extract_embedding(model, tokenizer, seq, device, SEQ_LEN, amp_dtype)
        if emb is None:
            continue
        embeddings.append(emb)
        labels.append(1 if is_human_virus(header) else 0)
        if (i + 1) % 50 == 0:
            print(f"    Embeddings: {i+1}/{len(records)}", flush=True)

    X = np.array(embeddings)
    y = np.array(labels)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"    Samples: {len(y)} (human={n_pos}, non-human={n_neg})", flush=True)

    if n_pos < 5 or n_neg < 5:
        print("    SKIP: not enough samples per class", flush=True)
        return {"task": "host_tropism", "checkpoint": ckpt_name, "error": "insufficient_samples"}

    # 5-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs, aucs, f1s = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        clf.fit(X_tr, y[train_idx])
        pred = clf.predict(X_te)
        prob = clf.predict_proba(X_te)[:, 1]
        accs.append(accuracy_score(y[test_idx], pred))
        try:
            aucs.append(roc_auc_score(y[test_idx], prob))
        except ValueError:
            aucs.append(float('nan'))
        f1s.append(f1_score(y[test_idx], pred))

    result = {
        "task": "host_tropism",
        "checkpoint": ckpt_name,
        "n_samples": len(y),
        "n_human": int(n_pos),
        "n_nonhuman": int(n_neg),
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "auroc": float(np.nanmean(aucs)),
        "auroc_std": float(np.nanstd(aucs)),
        "f1": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }
    print(f"    -> Accuracy={result['accuracy']:.3f}±{result['accuracy_std']:.3f}  "
          f"AUROC={result['auroc']:.3f}±{result['auroc_std']:.3f}  "
          f"F1={result['f1']:.3f}", flush=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TASK 2: CpG Suppression Correlation
# ═══════════════════════════════════════════════════════════════════════════

def compute_cpg_oe(seq: str) -> float:
    """CpG observed/expected ratio."""
    seq = seq.upper()
    n = len(seq)
    if n < 2:
        return float('nan')
    cpg = sum(1 for i in range(n-1) if seq[i:i+2] == "CG")
    c = seq.count("C")
    g = seq.count("G")
    expected = (c * g) / n if n > 0 else 0
    return cpg / expected if expected > 0 else float('nan')


def task_cpg_suppression(model, tokenizer, records, device, amp_dtype, ckpt_name):
    """Test if model's per-position log-probs correlate with local CpG density.

    For each sequence, compute:
    - Local CpG O/E in sliding windows (biological signal)
    - Average model log-prob in same windows
    Then measure Spearman correlation across windows within each sequence,
    and average across sequences.

    A biologically-aware model should assign lower probability (more negative
    log-probs) at positions within CpG-rich regions of human viruses, because
    those regions are under immune selection pressure.
    """
    print("\n  [TASK 2] CpG Suppression Correlation", flush=True)
    window = 100  # nt window for local CpG density

    human_corrs = []
    nonhuman_corrs = []

    for i, (header, seq) in enumerate(records):
        logprobs = get_per_position_logprobs(model, tokenizer, seq, device, SEQ_LEN, amp_dtype)
        if logprobs is None or len(logprobs) < window + 10:
            continue

        # Compute local CpG O/E in sliding windows aligned with logprob positions
        # logprobs[i] corresponds to position i+1 in the sequence
        n = min(len(logprobs), len(seq) - 1)
        cpg_windows = []
        lp_windows = []
        for start in range(0, n - window, window // 2):
            end = start + window
            if end > n:
                break
            subseq = seq[start+1:end+1]  # offset by 1 to align with logprob positions
            cpg_oe = compute_cpg_oe(subseq)
            if math.isnan(cpg_oe):
                continue
            mean_lp = float(logprobs[start:end].mean())
            cpg_windows.append(cpg_oe)
            lp_windows.append(mean_lp)

        if len(cpg_windows) < 5:
            continue

        rho, _ = spearmanr(cpg_windows, lp_windows)
        if math.isnan(rho):
            continue

        if is_human_virus(header):
            human_corrs.append(rho)
        else:
            nonhuman_corrs.append(rho)

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(records)}", flush=True)

    result = {
        "task": "cpg_suppression",
        "checkpoint": ckpt_name,
        "human_n": len(human_corrs),
        "human_mean_rho": float(np.mean(human_corrs)) if human_corrs else float('nan'),
        "human_std_rho": float(np.std(human_corrs)) if human_corrs else float('nan'),
        "nonhuman_n": len(nonhuman_corrs),
        "nonhuman_mean_rho": float(np.mean(nonhuman_corrs)) if nonhuman_corrs else float('nan'),
        "nonhuman_std_rho": float(np.std(nonhuman_corrs)) if nonhuman_corrs else float('nan'),
    }
    # A biologically-aware model should show NEGATIVE correlation for human viruses
    # (lower model probability where CpG is enriched => model "knows" CpG is avoided)
    print(f"    -> Human:     ρ={result['human_mean_rho']:+.4f}±{result['human_std_rho']:.4f} (n={result['human_n']})", flush=True)
    print(f"    -> Non-human: ρ={result['nonhuman_mean_rho']:+.4f}±{result['nonhuman_std_rho']:.4f} (n={result['nonhuman_n']})", flush=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# TASK 3: Codon Adaptation to Human Host
# ═══════════════════════════════════════════════════════════════════════════

# Human codon usage (per thousand, from Kazusa DB)
# Higher = more preferred in human
HUMAN_CODON_FREQ = {
    'TTT': 17.6, 'TTC': 20.3, 'TTA': 7.7, 'TTG': 12.9,
    'CTT': 13.2, 'CTC': 19.6, 'CTA': 7.2, 'CTG': 39.6,
    'ATT': 16.0, 'ATC': 20.8, 'ATA': 7.5, 'ATG': 22.0,
    'GTT': 11.0, 'GTC': 14.5, 'GTA': 7.1, 'GTG': 28.1,
    'TAT': 12.2, 'TAC': 15.3, 'TAA': 1.0, 'TAG': 0.8,
    'CAT': 10.9, 'CAC': 15.1, 'CAA': 12.3, 'CAG': 34.2,
    'AAT': 17.0, 'AAC': 19.1, 'AAA': 24.4, 'AAG': 31.9,
    'GAT': 21.8, 'GAC': 25.1, 'GAA': 29.0, 'GAG': 39.6,
    'TCT': 15.2, 'TCC': 17.7, 'TCA': 12.2, 'TCG': 4.4,
    'CCT': 17.5, 'CCC': 19.8, 'CCA': 16.9, 'CCG': 6.9,
    'ACT': 13.1, 'ACC': 18.9, 'ACA': 15.1, 'ACG': 6.1,
    'GCT': 18.4, 'GCC': 27.7, 'GCA': 15.8, 'GCG': 7.4,
    'TGT': 10.6, 'TGC': 12.6, 'TGA': 1.6, 'TGG': 13.2,
    'CGT': 4.5, 'CGC': 10.4, 'CGA': 6.2, 'CGG': 11.4,
    'AGT': 12.1, 'AGC': 19.5, 'AGA': 12.2, 'AGG': 12.0,
    'GGT': 10.8, 'GGC': 22.2, 'GGA': 16.5, 'GGG': 16.5,
}

# Group codons by amino acid (standard genetic code)
CODON_TABLE = {
    'F': ['TTT', 'TTC'], 'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'I': ['ATT', 'ATC', 'ATA'], 'M': ['ATG'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'Y': ['TAT', 'TAC'], '*': ['TAA', 'TAG', 'TGA'],
    'H': ['CAT', 'CAC'], 'Q': ['CAA', 'CAG'],
    'N': ['AAT', 'AAC'], 'K': ['AAA', 'AAG'],
    'D': ['GAT', 'GAC'], 'E': ['GAA', 'GAG'],
    'C': ['TGT', 'TGC'], 'W': ['TGG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
}

# Reverse: codon -> amino acid
CODON_TO_AA = {}
for aa, codons in CODON_TABLE.items():
    for c in codons:
        CODON_TO_AA[c] = aa

# For synonymous codons, compute relative adaptedness w_ij = freq_ij / max_freq_for_aa
CODON_W = {}
for aa, codons in CODON_TABLE.items():
    if len(codons) <= 1:
        for c in codons:
            CODON_W[c] = 1.0
        continue
    max_freq = max(HUMAN_CODON_FREQ.get(c, 0) for c in codons)
    for c in codons:
        CODON_W[c] = HUMAN_CODON_FREQ.get(c, 0) / max_freq if max_freq > 0 else 0


def compute_cai(seq: str) -> float:
    """Compute Codon Adaptation Index for human."""
    seq = seq.upper()
    log_w_sum = 0.0
    count = 0
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if codon not in CODON_W:
            continue
        aa = CODON_TO_AA.get(codon)
        if aa in ('M', 'W', '*'):  # only 1 codon or stop
            continue
        w = CODON_W[codon]
        if w > 0:
            log_w_sum += math.log(w)
            count += 1
    if count == 0:
        return float('nan')
    return math.exp(log_w_sum / count)


def task_codon_adaptation(model, tokenizer, records, device, amp_dtype, ckpt_name):
    """For each virus, compute CAI (human codon adaptation) and compare with
    model log-likelihood. More human-adapted viruses should get higher model
    log-likelihood if the model understands codon usage biology.

    We measure: Spearman correlation between per-sequence CAI and per-sequence
    mean log-probability across all virus sequences.
    """
    print("\n  [TASK 3] Codon Adaptation Correlation", flush=True)

    cai_scores = []
    model_lps = []
    is_human = []

    for i, (header, seq) in enumerate(records):
        # CAI on first 3000 nt (proxy for main ORF)
        cai = compute_cai(seq[:3000])
        if math.isnan(cai):
            continue

        logprobs = get_per_position_logprobs(model, tokenizer, seq, device, SEQ_LEN, amp_dtype)
        if logprobs is None or len(logprobs) < 50:
            continue

        mean_lp = float(logprobs.mean())
        cai_scores.append(cai)
        model_lps.append(mean_lp)
        is_human.append(is_human_virus(header))

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(records)}", flush=True)

    cai_arr = np.array(cai_scores)
    lp_arr = np.array(model_lps)
    human_mask = np.array(is_human)

    # Overall correlation
    rho_all, p_all = spearmanr(cai_arr, lp_arr)
    # Human viruses only
    if human_mask.sum() >= 5:
        rho_human, p_human = spearmanr(cai_arr[human_mask], lp_arr[human_mask])
    else:
        rho_human, p_human = float('nan'), float('nan')
    # Non-human
    if (~human_mask).sum() >= 5:
        rho_nonhuman, p_nonhuman = spearmanr(cai_arr[~human_mask], lp_arr[~human_mask])
    else:
        rho_nonhuman, p_nonhuman = float('nan'), float('nan')

    result = {
        "task": "codon_adaptation",
        "checkpoint": ckpt_name,
        "n_total": len(cai_scores),
        "n_human": int(human_mask.sum()),
        "rho_all": float(rho_all),
        "p_all": float(p_all),
        "rho_human": float(rho_human),
        "p_human": float(p_human) if not math.isnan(p_human) else None,
        "rho_nonhuman": float(rho_nonhuman),
        "p_nonhuman": float(p_nonhuman) if not math.isnan(p_nonhuman) else None,
        "mean_cai_human": float(cai_arr[human_mask].mean()) if human_mask.sum() > 0 else None,
        "mean_cai_nonhuman": float(cai_arr[~human_mask].mean()) if (~human_mask).sum() > 0 else None,
    }
    print(f"    -> All:       ρ={rho_all:+.4f} (p={p_all:.2e}, n={len(cai_scores)})", flush=True)
    print(f"    -> Human:     ρ={rho_human:+.4f} (n={int(human_mask.sum())})", flush=True)
    print(f"    -> Non-human: ρ={rho_nonhuman:+.4f} (n={int((~human_mask).sum())})", flush=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    set_seed(SEED)
    amp_dtype, _ = get_amp_settings()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all virus sequences with headers
    records = parse_fasta_with_headers("data/attack.fasta", min_len=512)
    print(f"Loaded {len(records)} virus sequences from attack.fasta", flush=True)

    human_count = sum(1 for h, _ in records if is_human_virus(h))
    print(f"  Human-infecting: {human_count}, Non-human: {len(records) - human_count}", flush=True)

    all_results = []
    out_jsonl = os.path.join(OUT_DIR, "downstream_results.jsonl")
    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    for ckpt_name, ckpt_path in CHECKPOINTS:
        print(f"\n{'='*70}", flush=True)
        print(f"Checkpoint: {ckpt_name}", flush=True)
        print(f"{'='*70}", flush=True)

        model, tokenizer = load_evo_model("evo-1-8k-base", DEVICE)
        maybe_load_locked_checkpoint(model, ckpt_path)
        model.eval()

        with specdef_fused_eval(model):
            # Task 1: Host tropism probe
            r1 = task_host_tropism(model, tokenizer, records, DEVICE, amp_dtype, ckpt_name)
            all_results.append(r1)
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(r1) + "\n")

            # Task 2: CpG suppression
            r2 = task_cpg_suppression(model, tokenizer, records, DEVICE, amp_dtype, ckpt_name)
            all_results.append(r2)
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(r2) + "\n")

            # Task 3: Codon adaptation
            r3 = task_codon_adaptation(model, tokenizer, records, DEVICE, amp_dtype, ckpt_name)
            all_results.append(r3)
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(r3) + "\n")

        del model
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}\n", flush=True)

    print("Task 1: Host Tropism Probe (5-fold CV)", flush=True)
    print(f"{'Checkpoint':20s}  {'Accuracy':>10}  {'AUROC':>10}  {'F1':>10}", flush=True)
    print("-" * 56, flush=True)
    for r in all_results:
        if r.get("task") == "host_tropism" and "error" not in r:
            print(f"{r['checkpoint']:20s}  {r['accuracy']:>10.3f}  "
                  f"{r['auroc']:>10.3f}  {r['f1']:>10.3f}", flush=True)

    print(f"\nTask 2: CpG Suppression Correlation (Spearman ρ)", flush=True)
    print(f"{'Checkpoint':20s}  {'Human ρ':>10}  {'NonHuman ρ':>12}", flush=True)
    print("-" * 46, flush=True)
    for r in all_results:
        if r.get("task") == "cpg_suppression":
            print(f"{r['checkpoint']:20s}  {r['human_mean_rho']:>+10.4f}  "
                  f"{r['nonhuman_mean_rho']:>+12.4f}", flush=True)

    print(f"\nTask 3: Codon Adaptation (Spearman ρ: CAI vs model log-prob)", flush=True)
    print(f"{'Checkpoint':20s}  {'All ρ':>10}  {'Human ρ':>10}  {'NonHuman ρ':>12}", flush=True)
    print("-" * 58, flush=True)
    for r in all_results:
        if r.get("task") == "codon_adaptation":
            print(f"{r['checkpoint']:20s}  {r['rho_all']:>+10.4f}  "
                  f"{r['rho_human']:>+10.4f}  {r['rho_nonhuman']:>+12.4f}", flush=True)

    print(f"\nResults saved to: {out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
