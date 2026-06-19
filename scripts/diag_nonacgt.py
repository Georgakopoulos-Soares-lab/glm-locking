"""Diagnose what non-ACGT bytes Evo checkpoints emit during prompted generation."""
import os, sys, collections
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.utils import load_evo_model, maybe_load_locked_checkpoint
from stripedhyena.generation import Generator
from Bio import SeqIO

CKPTS = {
    "pretrained": None,
    "M_a300k": "results/ft_locked_a300k_lr1e5_25k_locked/model_finetuned.pt",
}
model_name = sys.argv[1] if len(sys.argv) > 1 else "M_a300k"
PROMPT_FASTA = f"{REPO}/data/attack_heldout.fasta"

# first usable prompt
prompt_seq = None
for rec in SeqIO.parse(PROMPT_FASTA, "fasta"):
    s = "".join(c for c in str(rec.seq).upper() if c in "ACGT")
    if len(s) >= 512:
        prompt_seq = s[:512]; break

print(f"Loading {model_name}...")
model, tok = load_evo_model("evo-1-8k-base", "cuda")
maybe_load_locked_checkpoint(model, CKPTS[model_name])
model.eval()
gen = Generator(model, tok, top_k=0, top_p=0.95, temperature=0.8)

torch.manual_seed(42)
pids = tok.tokenize(prompt_seq)
if isinstance(pids, np.ndarray): pids = pids.tolist()
inp = torch.tensor([pids], dtype=torch.long, device="cuda")
with torch.no_grad():
    g, _ = gen.generate(device="cuda", input_ids=inp, num_tokens=2048,
                        cached_generation=True, print_generation=False,
                        verbose=False, stop_at_eos=False)
ids = g[0].tolist()
raw = tok.detokenize(ids)
print(f"\nraw len={len(raw)}  (n_tokens=2048)")
cnt = collections.Counter(raw)
print("char histogram (top 15):")
for ch, n in cnt.most_common(15):
    print(f"  {repr(ch)} (ord={ord(ch) if len(ch)==1 else '?'}): {n}  ({100*n/len(raw):.1f}%)")
acgt = sum(cnt[c] for c in "ACGT")
acgt_lower = sum(cnt[c] for c in "acgt")
print(f"\nACGT (upper): {acgt} ({100*acgt/len(raw):.1f}%)")
print(f"acgt (lower): {acgt_lower} ({100*acgt_lower/len(raw):.1f}%)")
print(f"N/n: {cnt.get('N',0)+cnt.get('n',0)}")
print(f"first 200 raw chars: {raw[:200]}")
