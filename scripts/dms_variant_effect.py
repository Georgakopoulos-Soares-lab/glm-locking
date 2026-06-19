"""Zero-shot DMS variant-effect probe — a composition-IMMUNE capability axis.

Why this exists (read me):
  Discriminative axes we have tried (HVUE AUROC, ViroBench host) are
  composition-confounded: a no-model k-mer baseline ties or beats every Evo
  checkpoint, so they cannot demonstrate sequence-function *understanding*.
  Deep-mutational-scanning (DMS) fixes this structurally. Within one assay every
  variant is the SAME gene differing by a single amino-acid substitution
  (≤3 nt of 603). Composition is essentially constant across variants, so a
  k-mer model is structurally near-useless — yet function (ACE2 binding,
  expression) varies a lot. If a model's *zero-shot* likelihood ranks variants
  by their measured effect, that is genuine, composition-immune capability.

Dataset:
  Starr et al. 2020 (Bloom lab) SARS-CoV-2 RBD single-mutant DMS.
  - functional scores: bind_avg (ACE2 binding), expr_avg (RBD expression).
  - parental nucleotide background: the EXACT PacBio amplicon Starr used
    (data/PacBio_amplicons.gb, LOCUS "SARS-CoV-2", gene 34..636 = RBD CDS,
    201 codons, residues 331-531). Verified: CDS translation matches all 201
    DMS wildtype residues exactly (0 mismatches).

Method (zero-shot, teacher-forced):
  * Reverse-translate each amino-acid mutant by replacing the WT codon with the
    MINIMAL-edit synonymous codon for the mutant residue (deterministic; ties by
    fixed codon order). This minimises spurious composition change.
  * LLR(variant) = logP_model(mutant CDS) − logP_model(WT CDS), summed over all
    603 token positions (autoregressive teacher forcing). Higher LLR = model
    finds the variant more plausible.
  * Capability metric = Spearman(LLR, functional score) over all ~4221 variants.
    A fit-preserving model should give POSITIVE correlation (deleterious
    mutations are both low-binding and low-likelihood).

Composition control (pre-committed, matched):
  * k-mer baseline: Ridge regression from the mutant CDS's k-mer frequency vector
    to the functional score, 5-fold CV; report Spearman of out-of-fold preds.
    Because single-nt variants barely move the k-mer vector this should be ≈0 —
    that is the structural point. We report model−kmer residual + bootstrap CI.

Checkpoints (same registry & locked-eval path as virobench_host_probe.py):
  pretrained / locked_no_ft / unlocked_ft / M_a300k. Locked checkpoints run the
  FLOAT64-fused bf16 path (fused_eval_f64) so the attack's weight changes are
  evaluated on a clean, matched path.

Usage:
  CUDA_VISIBLE_DEVICES=7 python scripts/dms_variant_effect.py \
      --ckpts pretrained unlocked_ft M_a300k locked_no_ft \
      --out experiments/exp4_dms/rbd_variant_effect.csv
"""
from __future__ import annotations
import os, sys, re, csv, time, argparse, itertools, contextlib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F
from src.utils import load_evo_model, maybe_load_locked_checkpoint, get_amp_settings
from scripts.virobench_host_probe import fused_eval_f64

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "experiments/exp4_dms/data"
DMS_CSV = f"{DATA_DIR}/starr_rbd_single_mut.csv"
GB_FILE = f"{DATA_DIR}/pacbio_amplicons.gb"
EMB_DIR = "experiments/exp4_dms/llr_cache"

RBD_FIRST_SITE = 331           # site_SARS2 of the first RBD codon
SCORE_COLS = ["bind_avg", "expr_avg"]
KMER_K = 4
N_BOOT = 5000
SEED = 0
BATCH = 16
NUCLEOTIDES = ['A', 'C', 'G', 'T']

CKPTS = {
    "pretrained":   None,
    "locked_no_ft": "results/lock_alpha300k/model_specdef.pt",
    "unlocked_ft":  "results/ft_unlocked_full910_25k_unlocked/model_finetuned.pt",
    "M_a300k":      "results/ft_locked_a300k_lr1e5_25k_locked/model_finetuned.pt",
}

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L',
    'CTA': 'L', 'CTG': 'L', 'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'TCT': 'S', 'TCC': 'S',
    'TCA': 'S', 'TCG': 'S', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'GCT': 'A', 'GCC': 'A',
    'GCA': 'A', 'GCG': 'A', 'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R',
    'CGA': 'R', 'CGG': 'R', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}
# amino acid -> list of synonymous codons (fixed order = sorted, deterministic)
AA2CODONS: dict[str, list[str]] = {}
for _c, _a in sorted(CODON_TABLE.items()):
    AA2CODONS.setdefault(_a, []).append(_c)


# ── Reference parsing ─────────────────────────────────────────────────────────

def parse_rbd_cds() -> str:
    """Extract the SARS-CoV-2 RBD CDS (603 nt) from the PacBio amplicon GenBank."""
    txt = open(GB_FILE).read()
    rec = [r for r in txt.split('//') if 'LOCUS       SARS-CoV-2 ' in r][0]
    m = re.search(r'gene\s+(\d+)\.\.(\d+)', rec)
    start, end = int(m.group(1)), int(m.group(2))
    origin = rec.split('ORIGIN')[1]
    seq = ''.join(re.findall(r'[acgtnACGTN]', origin.replace('\n', ' '))).upper()
    cds = seq[start - 1:end]
    assert len(cds) % 3 == 0, f"CDS not codon-aligned: {len(cds)}"
    return cds


def translate(cds: str) -> str:
    return ''.join(CODON_TABLE[cds[i:i + 3]] for i in range(0, len(cds), 3))


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def minimal_edit_codon(wt_codon: str, mut_aa: str) -> str:
    """Synonymous codon for mut_aa closest (Hamming) to wt_codon; deterministic."""
    cands = AA2CODONS[mut_aa]
    return min(cands, key=lambda c: (_hamming(wt_codon, c), c))


def build_variants() -> pd.DataFrame:
    """Return df with mutation, scores, and the full mutant CDS nucleotide string."""
    cds = parse_rbd_cds()
    prot = translate(cds)
    d = pd.read_csv(DMS_CSV)
    # keep real single missense mutants with at least one finite functional score
    d = d[d["wildtype"] != d["mutant"]].copy()
    d = d[d["mutant"] != "*"].copy()
    seqs, ok = [], []
    for _, r in d.iterrows():
        site = int(r["site_SARS2"]); ci = site - RBD_FIRST_SITE
        wt_codon = cds[ci * 3:ci * 3 + 3]
        if CODON_TABLE[wt_codon] != r["wildtype"]:
            ok.append(False); seqs.append(None); continue
        mut_codon = minimal_edit_codon(wt_codon, r["mutant"])
        mut_cds = cds[:ci * 3] + mut_codon + cds[ci * 3 + 3:]
        seqs.append(mut_cds); ok.append(True)
    d["mut_cds"] = seqs
    d = d[pd.Series(ok, index=d.index)].reset_index(drop=True)
    d.attrs["wt_cds"] = cds
    print(f"[data] RBD CDS={len(cds)}nt {len(prot)}aa; built {len(d)} single missense variants")
    return d


# ── Stage 1: GPU — zero-shot teacher-forced log-likelihoods (cached) ──────────

@torch.no_grad()
def seq_loglik(model, tok, seqs, device, amp_dtype) -> np.ndarray:
    """Total teacher-forced log-likelihood (sum over tokens) for equal-length seqs."""
    ids_list = [tok.tokenize(s) for s in seqs]
    L = len(ids_list[0])
    assert all(len(x) == L for x in ids_list), "variable-length batch"
    ids = torch.tensor(ids_list, dtype=torch.long, device=device)
    with torch.autocast(device_type='cuda', dtype=amp_dtype):
        logits, _ = model(ids)
    logits = logits.float()
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)          # predict tokens 1..L-1
    tgt = ids[:, 1:]                                          # (B, L-1)
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    return tok_logp.sum(dim=1).cpu().numpy()


def compute_llr(ckpt_name, ckpt_path, variants, device) -> str:
    out = f"{EMB_DIR}/{ckpt_name}.npz"
    if os.path.exists(out):
        print(f"[llr] skip exists: {out}")
        return out
    os.makedirs(EMB_DIR, exist_ok=True)
    amp_dtype, _ = get_amp_settings()
    model, tok = load_evo_model('evo-1-8k-base', device)
    is_locked = bool(ckpt_path) and ("locked" in ckpt_name or "M_" in ckpt_name)
    maybe_load_locked_checkpoint(model, ckpt_path if ckpt_path else None)
    model.eval()

    wt_cds = variants.attrs["wt_cds"]
    seqs = variants["mut_cds"].tolist()
    ctx = fused_eval_f64(model) if is_locked else contextlib.nullcontext()
    with ctx:
        wt_ll = float(seq_loglik(model, tok, [wt_cds], device, amp_dtype)[0])
        lls = np.empty(len(seqs), dtype=np.float64)
        t0 = time.time()
        for i in range(0, len(seqs), BATCH):
            batch = seqs[i:i + BATCH]
            try:
                lls[i:i + len(batch)] = seq_loglik(model, tok, batch, device, amp_dtype)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                for j, s in enumerate(batch):
                    lls[i + j] = seq_loglik(model, tok, [s], device, amp_dtype)[0]
            if (i // BATCH) % 25 == 0 and i > 0:
                rate = (i + BATCH) / (time.time() - t0)
                print(f"   [{ckpt_name}] {i + BATCH}/{len(seqs)} ({rate:.1f} seq/s)")
    llr = lls - wt_ll
    np.savez_compressed(out, llr=llr, wt_ll=wt_ll, fused=is_locked)
    del model
    torch.cuda.empty_cache()
    print(f"[llr] wrote {out}  (wt_ll={wt_ll:.2f}, mean LLR={llr.mean():.3f})")
    return out


# ── k-mer composition baseline (matched, pre-committed) ───────────────────────

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


def kmer_cv_pred(X, y) -> np.ndarray:
    """Out-of-fold Ridge predictions (5-fold), alpha by inner CV.

    NOTE: used ONLY for the supervised-additive REFERENCE, not as the composition
    control. A supervised regression on per-variant k-mer vectors exploits the
    k-mer fingerprint as a (lossy) one-hot of the mutation's (site, identity) and
    therefore MEMORISES the DMS landscape from the labels — it is an "additive
    model" upper bound, not a composition statistic. The composition control is
    the UNSUPERVISED kmer-L1-distance-from-WT (comp_null_stat below), which is ~0.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    pred = np.zeros_like(y, dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        m.fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return pred


def comp_null_stat(seqs, wt_cds, k: int) -> np.ndarray:
    """UNSUPERVISED composition null: L1 distance of each variant's k-mer freq
    vector from the WT's. This sees NO labels and is the structural composition
    control — for single-nt variants it barely moves, so its correlation with the
    functional score should be ≈0 (composition immunity)."""
    Xk = kmer_frequencies(seqs, k)
    xwt = kmer_frequencies([wt_cds], k)[0]
    return np.abs(Xk - xwt).sum(axis=1)


# ── Stats ─────────────────────────────────────────────────────────────────────

def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def boot_spearman_ci(pred, score, n_boot=N_BOOT):
    """Bootstrap CI for a single Spearman(pred, score)."""
    rng = np.random.default_rng(SEED)
    n = len(score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = spearman(pred[idx], score[idx])
        if not np.isnan(r):
            vals.append(r)
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_residual_ci(score, model_val, null_stat, n_boot=N_BOOT):
    """Paired bootstrap: residual = |Spearman(model,score)| − |Spearman(null,score)|.

    Uses |·| because the unsupervised composition null can correlate with either
    sign; capability is "explains more rank-variance than composition alone."
    """
    rng = np.random.default_rng(SEED)
    n = len(score)
    res = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = spearman(model_val[idx], score[idx])
        k = spearman(null_stat[idx], score[idx])
        if not (np.isnan(m) or np.isnan(k)):
            res.append(abs(m) - abs(k))
    res = np.array(res)
    return float(np.mean(res)), float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


def boot_delta_ci(score, val_a, val_b, n_boot=N_BOOT):
    """Paired bootstrap of Δρ = Spearman(a,score) − Spearman(b,score) (a,b = two
    checkpoints' LLRs on the SAME variants). Used for path-artifact / FT / locking
    deltas, exactly mirroring the generative analysis."""
    rng = np.random.default_rng(SEED)
    n = len(score)
    res = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ra = spearman(val_a[idx], score[idx])
        rb = spearman(val_b[idx], score[idx])
        if not (np.isnan(ra) or np.isnan(rb)):
            res.append(ra - rb)
    res = np.array(res)
    return float(np.mean(res)), float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', default=list(CKPTS.keys()))
    ap.add_argument('--kmer_k', type=int, default=KMER_K)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='experiments/exp4_dms/rbd_variant_effect.csv')
    ap.add_argument('--extract_only', action='store_true')
    ap.add_argument('--additive_ref', action='store_true',
                    help='also compute the supervised k-mer additive reference (slow, context only)')
    a = ap.parse_args()

    variants = build_variants()

    # Stage 1: GPU LLR per checkpoint (cached)
    for name in a.ckpts:
        if name not in CKPTS:
            print(f"[warn] unknown ckpt '{name}', skipping"); continue
        compute_llr(name, CKPTS[name], variants, a.device)
    if a.extract_only:
        print("extract_only: done"); return

    wt_cds = variants.attrs["wt_cds"]
    # UNSUPERVISED composition null on the SAME variant CDS sequences
    print(f"\n[null] unsupervised k={a.kmer_k} composition null (kmer-L1-from-WT)...")
    null_full = comp_null_stat(variants["mut_cds"].tolist(), wt_cds, a.kmer_k)
    # optional supervised additive reference (sees labels; NOT a composition control)
    Xk = kmer_frequencies(variants["mut_cds"].tolist(), a.kmer_k) if a.additive_ref else None

    # cache LLRs for the pairwise deltas
    llrs = {}
    for name in a.ckpts:
        npz = f"{EMB_DIR}/{name}.npz"
        if os.path.exists(npz):
            llrs[name] = np.load(npz, allow_pickle=True)["llr"]

    rows = []
    for name in a.ckpts:
        npz = f"{EMB_DIR}/{name}.npz"
        if not os.path.exists(npz):
            print(f"[warn] no LLR for {name}; skipping"); continue
        llr = np.load(npz, allow_pickle=True)["llr"]
        fused = bool(np.load(npz, allow_pickle=True)["fused"])
        for col in SCORE_COLS:
            mask = variants[col].notna().values
            y = variants.loc[mask, col].values.astype(float)
            mval = llr[mask]
            nstat = null_full[mask]
            m_rho, m_lo, m_hi = boot_spearman_ci(mval, y)            # zero-shot capability
            null_rho = spearman(nstat, y)                           # unsupervised composition null
            rmean, rlo, rhi = boot_residual_ci(y, mval, nstat)      # |model| − |null|
            beats = "yes" if (rlo > 0 and m_lo > 0) else ("neg" if rhi < 0 else "ns")
            row = dict(ckpt=name, fused=fused, score=col, n=int(mask.sum()),
                       model_spearman=round(m_rho, 4), model_ci_lo=round(m_lo, 4),
                       model_ci_hi=round(m_hi, 4), comp_null_spearman=round(null_rho, 4),
                       residual_abs=round(rmean, 4), res_ci_lo=round(rlo, 4),
                       res_ci_hi=round(rhi, 4), beats_null=beats, kmer_k=a.kmer_k)
            if a.additive_ref:
                row["supervised_additive_spearman"] = round(spearman(kmer_cv_pred(Xk[mask], y), y), 4)
            rows.append(row)
            print(f"[probe] {name:13s} {col:9s} model_ρ={m_rho:+.4f} [{m_lo:+.4f},{m_hi:+.4f}] "
                  f"null_ρ={null_rho:+.4f} |Δ|={rmean:+.4f} [{rlo:+.4f},{rhi:+.4f}] beats_null={beats}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[done] wrote {a.out}")

    # ── Pairwise model-vs-model deltas (mirror the generative analysis) ──────
    pair_defs = [("path_artifact", "locked_no_ft", "pretrained"),
                 ("ft_effect", "unlocked_ft", "pretrained"),
                 ("lock_effect", "M_a300k", "unlocked_ft")]
    have = set(llrs)
    drows = []
    print("\n=== pairwise Δ Spearman (paired bootstrap) ===")
    for col in SCORE_COLS:
        mask = variants[col].notna().values
        y = variants.loc[mask, col].values.astype(float)
        for label, an, bn in pair_defs:
            if an not in have or bn not in have:
                continue
            dm, dlo, dhi = boot_delta_ci(y, llrs[an][mask], llrs[bn][mask])
            sig = "sig" if (dlo > 0 or dhi < 0) else "ns"
            drows.append(dict(comparison=label, a=an, b=bn, score=col,
                              delta_spearman=round(dm, 4), ci_lo=round(dlo, 4),
                              ci_hi=round(dhi, 4), significant=sig))
            print(f"  {label:14s} {col:9s} Δρ({an}−{bn})={dm:+.4f} [{dlo:+.4f},{dhi:+.4f}] {sig}")
    dout = a.out.replace('.csv', '_pairwise.csv')
    with open(dout, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(drows[0].keys()))
        w.writeheader(); w.writerows(drows)
    print(f"[done] wrote {dout}")


if __name__ == '__main__':
    main()
