"""Compare biological modeling capability across locked vs unlocked checkpoints.

Evaluates per-sequence perplexity on human virus genomes to measure
whether the SpecDef lock degrades the attacker's ability to model
dangerous biological sequences.

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -n evo --no-capture-output \
        python -u scripts/eval_bio_compare.py
"""

import os
import sys
import csv
import math
import re
from contextlib import contextmanager, nullcontext

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    set_seed,
    get_amp_settings,
    load_evo_model,
    maybe_load_locked_checkpoint,
    maybe_enable_gradient_checkpointing,
    specdef_fused_eval,
)


# ===========================================================================
# Config
# ===========================================================================
DEVICE = "cuda:0"
FASTA_PATH = "data/attack.fasta"
RESULTS_DIR = "results/eval_bio_compare"
SEQ_LEN = 1024
SEED = 42

# Checkpoints to compare
CHECKPOINTS = {
    "pretrained":       None,
    "unlocked_lr1e5":   "results/ft_paper_lr1e5_unlocked/model_finetuned.pt",
    "locked_lr1e5":     "results/ft_paper_lr1e5_locked/model_finetuned.pt",
    "locked_lr3e5":     "results/ft_paper_lr3e5_locked/model_finetuned.pt",
}

# Human virus patterns to filter
HUMAN_VIRUS_PATTERNS = [
    r"[Hh]uman", r"HIV", r"SARS", r"[Hh]epatitis", r"[Ii]nfluenza",
    r"[Ee]bola", r"[Cc]oronavirus", r"HPV",
]


# ===========================================================================
# Helpers
# ===========================================================================

def load_fasta(path):
    """Load FASTA → list of (header, sequence)."""
    entries = []
    header, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        entries.append((header, "".join(seq_parts)))
    return entries


def is_human_virus(header):
    for pat in HUMAN_VIRUS_PATTERNS:
        if re.search(pat, header):
            return True
    return False


def per_sequence_ppl(model, tokenizer, sequence, device, seq_len, amp_dtype):
    """Compute perplexity on a single sequence using sliding windows."""
    tok_ids = torch.tensor(tokenizer.tokenize(sequence), dtype=torch.long)
    if len(tok_ids) < 10:
        return None

    losses = []
    n_windows = max(1, (len(tok_ids) - 1) // seq_len)
    for i in range(n_windows):
        start = i * seq_len
        end = min(start + seq_len + 1, len(tok_ids))
        chunk = tok_ids[start:end].unsqueeze(0).to(device)
        if chunk.shape[1] < 10:
            continue
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = model(chunk[:, :-1])
            targets = chunk[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="mean",
            )
        losses.append(loss.item())

    if not losses:
        return None
    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)


def evaluate_checkpoint(ckpt_name, ckpt_path, human_seqs, all_virus_seqs,
                        retain_seqs, device, seq_len, amp_dtype):
    """Load model, evaluate on sequences, return results dict."""
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluating: {ckpt_name}", flush=True)
    print(f"  checkpoint: {ckpt_path or 'pretrained (no ckpt)'}", flush=True)
    print(f"{'='*60}", flush=True)

    model, tokenizer = load_evo_model("evo-1-8k-base", device)

    if ckpt_path:
        maybe_load_locked_checkpoint(model, ckpt_path)

    maybe_enable_gradient_checkpointing(model, True)
    model.eval()

    results = {"checkpoint": ckpt_name}

    # Use fused eval context for SpecDef models (no-op for plain models)
    with specdef_fused_eval(model):
        # Evaluate on human virus sequences
        print(f"\n  Evaluating on {len(human_seqs)} human virus sequences...", flush=True)
        human_ppls = []
        for i, (header, seq) in enumerate(human_seqs):
            ppl = per_sequence_ppl(model, tokenizer, seq, device, seq_len, amp_dtype)
            if ppl is not None:
                human_ppls.append(ppl)
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(human_seqs)}]", flush=True)

        results["human_virus_ppl_mean"] = sum(human_ppls) / len(human_ppls) if human_ppls else None
        results["human_virus_ppl_median"] = sorted(human_ppls)[len(human_ppls)//2] if human_ppls else None
        results["human_virus_n"] = len(human_ppls)
        print(f"    mean PPL: {results['human_virus_ppl_mean']:.3f}  "
              f"median: {results['human_virus_ppl_median']:.3f}  "
              f"(n={results['human_virus_n']})", flush=True)

        # Evaluate on ALL virus sequences (sample for speed)
        sample_all = all_virus_seqs[:100]
        print(f"  Evaluating on {len(sample_all)} virus sequences (sample)...", flush=True)
        all_ppls = []
        for i, (header, seq) in enumerate(sample_all):
            ppl = per_sequence_ppl(model, tokenizer, seq, device, seq_len, amp_dtype)
            if ppl is not None:
                all_ppls.append(ppl)
            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{len(sample_all)}]", flush=True)

        results["all_virus_ppl_mean"] = sum(all_ppls) / len(all_ppls) if all_ppls else None
        results["all_virus_ppl_median"] = sorted(all_ppls)[len(all_ppls)//2] if all_ppls else None
        results["all_virus_n"] = len(all_ppls)
        print(f"    mean PPL: {results['all_virus_ppl_mean']:.3f}  "
              f"median: {results['all_virus_ppl_median']:.3f}  "
              f"(n={results['all_virus_n']})", flush=True)

        # Evaluate on retain data (bacteria — should be preserved)
        sample_retain = retain_seqs[:50]
        print(f"  Evaluating on {len(sample_retain)} retain sequences (bacteria)...", flush=True)
        retain_ppls = []
        for i, (header, seq) in enumerate(sample_retain):
            ppl = per_sequence_ppl(model, tokenizer, seq, device, seq_len, amp_dtype)
            if ppl is not None:
                retain_ppls.append(ppl)
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(sample_retain)}]", flush=True)

        results["retain_ppl_mean"] = sum(retain_ppls) / len(retain_ppls) if retain_ppls else None
        results["retain_ppl_median"] = sorted(retain_ppls)[len(retain_ppls)//2] if retain_ppls else None
        results["retain_n"] = len(retain_ppls)
        print(f"    mean PPL: {results['retain_ppl_mean']:.3f}  "
              f"median: {results['retain_ppl_median']:.3f}  "
              f"(n={results['retain_n']})", flush=True)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return results, human_ppls


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)
    amp_dtype, _ = get_amp_settings()

    # Load all sequences
    print("Loading FASTA data...", flush=True)
    all_virus = load_fasta(FASTA_PATH)
    retain_all = load_fasta("data/retain.fasta")

    # Filter human viruses
    human_virus = [(h, s) for h, s in all_virus if is_human_virus(h)]
    print(f"  Total virus sequences: {len(all_virus)}")
    print(f"  Human virus sequences: {len(human_virus)}")
    print(f"  Retain sequences:      {len(retain_all)}")

    # List human viruses
    print("\nHuman virus sequences:")
    for h, s in human_virus:
        print(f"  {h[:80]}  ({len(s):,} bp)")

    # Evaluate each checkpoint
    all_results = []
    per_seq_data = {}

    for name, path in CHECKPOINTS.items():
        if path and not os.path.exists(path):
            print(f"\n  SKIP {name}: checkpoint not found at {path}")
            continue
        res, ppls = evaluate_checkpoint(
            name, path, human_virus, all_virus, retain_all,
            DEVICE, SEQ_LEN, amp_dtype,
        )
        all_results.append(res)
        per_seq_data[name] = ppls

        # Save incrementally after each checkpoint
        csv_path = os.path.join(RESULTS_DIR, "bio_compare.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Checkpoint':>20s}  {'HumanVirus':>12s}  {'AllVirus':>12s}  {'Retain':>12s}")
    print(f"{'':>20s}  {'mean PPL':>12s}  {'mean PPL':>12s}  {'mean PPL':>12s}")
    print("-"*62)
    for res in all_results:
        hv = res['human_virus_ppl_mean']
        av = res['all_virus_ppl_mean']
        rt = res['retain_ppl_mean']
        print(f"{res['checkpoint']:>20s}  "
              f"{hv:>12.3f}  {av:>12.3f}  {rt:>12.3f}")

    # Per-sequence human virus PPL comparison
    if len(per_seq_data) > 1:
        per_seq_path = os.path.join(RESULTS_DIR, "human_virus_per_seq.csv")
        with open(per_seq_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence"] + list(per_seq_data.keys()))
            for i, (h, _) in enumerate(human_virus):
                row = [h[:80]]
                for name in per_seq_data:
                    ppls = per_seq_data[name]
                    row.append(f"{ppls[i]:.3f}" if i < len(ppls) else "")
                w.writerow(row)
        print(f"Saved: {per_seq_path}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
