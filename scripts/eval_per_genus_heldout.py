"""Per-genus held-out attack PPL analysis (Test #10).

Uses existing data/attack_heldout.fasta (134 held-out genera, never seen at train).
For each available checkpoint, compute per-sequence PPL, group by genus,
report distribution and tail behavior.

Run after FT runs finish; works on any model checkpoint compatible with eval_attack_ppl.

Output:
  results/phase5_per_genus_heldout.csv
  results/phase5_per_genus_heldout.png  (boxplot per ckpt across genera)
"""
import os, sys, json, argparse, math
from collections import defaultdict
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.utils import load_evo_model, maybe_load_locked_checkpoint, get_amp_settings


def parse_fasta_with_genus(path, min_len=512):
    recs = []; cur_h = None; cur = []
    def flush():
        if cur_h is not None and cur:
            seq = "".join(cur).upper()
            if len(seq) >= min_len:
                # genus = first whitespace token after '>'
                tok = cur_h.lstrip(">").split()
                genus = tok[1] if len(tok) > 1 else "unknown"
                recs.append((cur_h, seq, genus))
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                flush(); cur_h = line; cur = []
            else:
                cur.append(line)
    flush()
    return recs


def seq_ppl(model, tokenizer, seq, device, amp_dtype, max_len=4096):
    seq = seq[:max_len]
    ids = torch.tensor([tokenizer.tokenize(seq)], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        out = model(ids)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out.logits
        logits = logits.float()
    tgt = ids[:, 1:].reshape(-1)
    lp = torch.log_softmax(logits[:, :-1], dim=-1).reshape(-1, logits.size(-1))
    nll = -lp[torch.arange(tgt.size(0)), tgt].mean().item()
    return math.exp(nll)


CHECKPOINTS = [
    ("pretrained", None),
    ("locked_10k", "results/lock_specdef_paper_all/model_specdef.pt"),
    ("locked_100k", "results/paper_lock_alpha100k/model_specdef.pt"),
    ("locked_1M", "results/paper_lock_alpha1M/model_specdef.pt"),
    ("ft_unlocked", "results/paper_ft_unlocked_unlocked/model_best.pt"),
    ("ft_locked_10k", "results/paper_ft_locked_10k_locked/model_best.pt"),
    ("fused_bypass_30k", "results/phase5_fused/fused_alpha30k.pt"),
    ("fused_bypass_1M", "results/phase5_fused/fused_alpha1M.pt"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", default="data/attack_heldout.fasta")
    p.add_argument("--out_csv", default="results/phase5_per_genus_heldout.csv")
    p.add_argument("--out_png", default="results/phase5_per_genus_heldout.png")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_per_genus", type=int, default=5)
    p.add_argument("--checkpoints", nargs="*", default=None)
    a = p.parse_args()

    recs = parse_fasta_with_genus(a.fasta)
    by_genus = defaultdict(list)
    for h, s, g in recs:
        by_genus[g].append((h, s))
    # cap per-genus to balance
    for g in by_genus:
        by_genus[g] = by_genus[g][:a.max_per_genus]
    n_eval = sum(len(v) for v in by_genus.values())
    print(f"Held-out: {len(by_genus)} genera, {n_eval} sequences (≤{a.max_per_genus}/genus)")

    amp_dtype, _ = get_amp_settings()
    rows = []
    cps = [c for c in CHECKPOINTS if (a.checkpoints is None or c[0] in a.checkpoints)]
    for name, ckpt in cps:
        if ckpt is not None and not os.path.exists(ckpt):
            print(f"skip {name} (missing {ckpt})"); continue
        print(f"\n=== {name} ===")
        model, tok = load_evo_model("evo-1-8k-base", a.device)
        if ckpt: maybe_load_locked_checkpoint(model, ckpt)
        model.eval()
        for g, seqs in by_genus.items():
            for h, s in seqs:
                try:
                    p_ = seq_ppl(model, tok, s, a.device, amp_dtype)
                    rows.append((name, g, h.split()[0].lstrip(">"), p_))
                except Exception as e:
                    print(f"  err {h[:30]}: {e}")
        del model, tok; torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    with open(a.out_csv, "w") as f:
        f.write("ckpt,genus,seq_id,ppl\n")
        for r in rows: f.write(",".join(map(str,r)) + "\n")
    print(f"\nwrote {a.out_csv}  ({len(rows)} rows)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.DataFrame(rows, columns=["ckpt","genus","seq_id","ppl"])
        # boxplot per ckpt over genus-mean PPLs
        gdf = df.groupby(["ckpt","genus"])["ppl"].mean().reset_index()
        order = sorted(gdf["ckpt"].unique())
        data = [gdf[gdf.ckpt==c]["ppl"].values for c in order]
        fig, ax = plt.subplots(figsize=(10,5))
        ax.boxplot(data, labels=order, showfliers=True)
        ax.set_ylabel("Per-genus mean PPL  (held-out viruses)")
        ax.set_title("Held-out viral generalization across 134 genera")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout(); plt.savefig(a.out_png, dpi=150)
        print(f"wrote {a.out_png}")
    except Exception as e:
        print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
