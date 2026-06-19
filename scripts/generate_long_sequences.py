"""Long-sequence PROMPTED generation (4096 nt continuation) for gene-content analysis.

WHY PROMPTED, NOT UNCONDITIONAL
--------------------------------
Unconditional generation at 4096 nt collapses into homopolymer repeats
(validated: GC=0.02 / 0.97, poly-T / poly-G runs) -- Evo loses long-range
coherence without a seed. Degenerate sequences contain zero genes regardless
of checkpoint, so the M-vs-unlocked_ft comparison would be meaningless.

PROTOCOL
--------
* Prompt  : first PROMPT_LEN (512) ACGT nt of a HELD-OUT viral genome
            (data/attack_heldout.fasta -- NOT in the FT corpus, avoids the
            memorization confound).
* Generate: MAX_NEW_TOKENS (4096) continuation tokens.
* Output  : the GENERATED continuation only (prompt stripped) so geNomad scores
            what the model produced, not the natural seed.
* Matched : identical prompts + per-seq seeds across all 4 checkpoints ->
            paired comparison. Locked checkpoints use the REAL SpecDef path.
* Sampling: temperature=0.8, top_p=0.95 (matched to the 513 nt run).
* Speed   : StripedHyena cached Generator (~40 tok/s).

Usage (one model per GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_long_sequences.py --model pretrained
  CUDA_VISIBLE_DEVICES=1 python scripts/generate_long_sequences.py --model locked_no_ft
  CUDA_VISIBLE_DEVICES=2 python scripts/generate_long_sequences.py --model unlocked_ft
  CUDA_VISIBLE_DEVICES=4 python scripts/generate_long_sequences.py --model M_a300k
"""
from __future__ import annotations
import os, sys, time, argparse, random
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.utils import load_evo_model, maybe_load_locked_checkpoint
from stripedhyena.generation import Generator

# -- Config (matched to scripts/generative_comparison.py where applicable) -----
N_SEQUENCES    = 100
PROMPT_LEN     = 512
MAX_NEW_TOKENS = 4096
TEMPERATURE    = 0.8
TOP_P          = 0.95
BASE_SEED      = 42
DEVICE         = "cuda"

CKPTS = {
    "pretrained":   None,
    "locked_no_ft": "results/lock_alpha300k/model_specdef.pt",
    "unlocked_ft":  "results/ft_unlocked_full910_25k_unlocked/model_finetuned.pt",
    "M_a300k":      "results/ft_locked_a300k_lr1e5_25k_locked/model_finetuned.pt",
}

PROMPT_FASTA = f"{REPO}/data/attack_heldout.fasta"
OUT_DIR      = f"{REPO}/experiments/exp3_virobench/generative_4096/sequences"
PROMPT_OUT   = f"{REPO}/experiments/exp3_virobench/generative_4096/prompts_used.fasta"


def load_prompts(n: int, prompt_len: int) -> list[tuple[str, str]]:
    """Return [(genome_id, first prompt_len ACGT nt), ...] for the first n usable
    held-out genomes. Deterministic order so all models share identical prompts."""
    from Bio import SeqIO
    prompts = []
    for rec in SeqIO.parse(PROMPT_FASTA, "fasta"):
        seq = "".join(c for c in str(rec.seq).upper() if c in "ACGT")
        if len(seq) >= prompt_len:
            prompts.append((rec.id, seq[:prompt_len]))
        if len(prompts) >= n:
            break
    if len(prompts) < n:
        raise RuntimeError(f"only {len(prompts)} usable prompts, need {n}")
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CKPTS.keys()))
    ap.add_argument("--n",    type=int, default=N_SEQUENCES)
    ap.add_argument("--ntok", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--prompt_len", type=int, default=PROMPT_LEN)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_fa = f"{OUT_DIR}/{a.model}.fasta"

    # Resume: count already-written records
    done = 0
    if os.path.exists(out_fa):
        with open(out_fa) as fh:
            done = sum(1 for ln in fh if ln.startswith(">"))
    if done >= a.n:
        print(f"[skip] {out_fa} already has {done} >= {a.n} sequences")
        return

    prompts = load_prompts(a.n, a.prompt_len)
    # Persist prompts once (so scoring/repro can reference the seeds)
    if not os.path.exists(PROMPT_OUT):
        with open(PROMPT_OUT, "w") as pf:
            for i, (gid, pseq) in enumerate(prompts):
                pf.write(f">prompt_{i}__{gid}\n{pseq}\n")
        print(f"[prompts] wrote {len(prompts)} seeds -> {PROMPT_OUT}")

    print(f"Loading model '{a.model}' (ckpt={CKPTS[a.model]})...")
    model, tok = load_evo_model("evo-1-8k-base", DEVICE)
    maybe_load_locked_checkpoint(model, CKPTS[a.model])
    model.eval()

    gen = Generator(model, tok, top_k=0, top_p=TOP_P, temperature=TEMPERATURE)

    fh = open(out_fa, "a")
    t0 = time.time()
    for i in range(done, a.n):
        gid, prompt_seq = prompts[i]
        torch.manual_seed(BASE_SEED + i)
        random.seed(BASE_SEED + i)

        prompt_ids = tok.tokenize(prompt_seq)
        if isinstance(prompt_ids, np.ndarray):
            prompt_ids = prompt_ids.tolist()
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            generation, _ = gen.generate(
                device=DEVICE,
                input_ids=input_ids,
                num_tokens=a.ntok,
                cached_generation=True,
                print_generation=False,
                verbose=False,
                stop_at_eos=False,
            )
        # generation contains ONLY the new tokens (prompt is not included)
        gen_ids = generation[0].tolist()
        raw = tok.detokenize(gen_ids) if hasattr(tok, "detokenize") else "".join(
            chr(c) for c in gen_ids)
        raw = raw.upper()
        # Map every non-ACGT byte -> N. Evo emits spaces / IUPAC ambiguity / junk
        # tokens when generation degrades (esp. locked-FT). N-masking KEEPS the
        # full length and positional register (no length confound, no frameshift
        # from concatenation), and geNomad/prodigal naturally break ORFs across
        # N-runs -> degraded generation shows up as genuinely missing/broken genes.
        seq = "".join(c if c in "ACGT" else "N" for c in raw)
        n_nonacgt = sum(1 for c in seq if c == "N")
        frac_nonacgt = n_nonacgt / max(len(seq), 1)

        # non-ACGT fraction is a length- and composition-immune degradation metric.
        fh.write(f">{a.model}_{i}__seed_{gid}__nonacgt{frac_nonacgt:.4f}\n{seq}\n")
        fh.flush()

        if (i + 1) % 5 == 0:
            el = time.time() - t0
            rate = (i + 1 - done) / el
            eta = (a.n - i - 1) / rate / 60 if rate > 0 else 0
            acgt = [c for c in seq if c in "ACGT"]
            gc = sum(c in "GC" for c in acgt) / max(len(acgt), 1)
            print(f"  {i+1}/{a.n}  len={len(seq)}  nonACGT={frac_nonacgt:.3f}  "
                  f"GC={gc:.3f}  {el:.0f}s  {rate*a.ntok:.0f} tok/s  ETA {eta:.0f} min",
                  flush=True)
    fh.close()
    print(f"[done] wrote {a.n} sequences -> {out_fa}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
