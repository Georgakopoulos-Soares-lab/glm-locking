"""Smoke test: cached vs naive generation speed + SpecDef compatibility.

Confirms the StripedHyena cached Generator works with our load_evo_model output
and (optionally) a SpecDef-locked checkpoint, and measures tokens/sec at length.
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils import load_evo_model, maybe_load_locked_checkpoint
from stripedhyena.generation import Generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="locked checkpoint path or None")
    ap.add_argument("--ntok", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    print(f"Loading model (ckpt={a.ckpt})...")
    model, tok = load_evo_model("evo-1-8k-base", a.device)
    maybe_load_locked_checkpoint(model, a.ckpt)
    model.eval()

    gen = Generator(model, tok, top_k=0, top_p=0.95, temperature=0.8)

    # seed prompt: single nucleotide so generation is (near) unconditional
    prompt_ids = torch.tensor([[tok.tokenize("A")[0] if hasattr(tok, "tokenize") else 65]],
                              dtype=torch.long, device=a.device)

    torch.manual_seed(0)
    t0 = time.time()
    with torch.no_grad():
        out = gen.generate(
            device=a.device,
            input_ids=prompt_ids,
            num_tokens=a.ntok,
            cached_generation=True,
            print_generation=False,
            verbose=False,
            stop_at_eos=False,
        )
    dt = time.time() - t0

    # out may be (ids, scores) or string depending on version; handle both
    if isinstance(out, tuple):
        ids = out[0]
        seq = tok.detokenize(ids[0].tolist()) if hasattr(tok, "detokenize") else str(ids.shape)
    else:
        seq = out
    seq_str = seq if isinstance(seq, str) else str(seq)
    print(f"[cached] {a.ntok} tokens in {dt:.1f}s = {a.ntok/dt:.1f} tok/s")
    print(f"  len(seq)={len(seq_str)}  head={seq_str[:60]!r}")
    print(f"  est for 4096 tok × 50 seqs = {4096/ (a.ntok/dt) * 50 / 60:.1f} min/model")


if __name__ == "__main__":
    main()
