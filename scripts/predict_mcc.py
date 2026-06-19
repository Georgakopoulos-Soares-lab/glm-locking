"""Compute MCC from saved LoRA weights (logit >= 0 threshold).
Loads saved LoRA+head state, runs inference on the val set, reports MCC.
"""
from __future__ import annotations
import os, sys, argparse, glob
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.utils import load_evo_model, maybe_load_locked_checkpoint
from src.lora import inject_lora
from scripts.hvue_rigorous_compare import repro_sample, N_TRAIN, N_VAL

LORA_RANK = 16
LORA_ALPHA = 32.0
LORA_TARGETS = ["projections", "out_filter_dense", "mlp.l1", "mlp.l2", "mlp.l3"]
MAX_LEN = 1000
SEED = 0
DEVICE = "cuda"
AMP_DTYPE = torch.bfloat16

CKPTS = {
    "pretrained":   None,
    "locked_no_ft": "results/lock_alpha300k/model_specdef.pt",
    "unlocked_ft":  "results/ft_unlocked_full910_25k_unlocked/model_finetuned.pt",
    "M_a300k":      "results/ft_locked_a300k_lr1e5_25k_locked/model_finetuned.pt",
}


class HVUEDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.seqs = df["sequence"].str.slice(0, MAX_LEN).tolist()
        self.labels = torch.tensor(df["label"].values.astype(np.float32))
        self._ids = [torch.tensor(tokenizer.tokenize(s), dtype=torch.long) for s in self.seqs]
    def __len__(self): return len(self._ids)
    def __getitem__(self, i): return self._ids[i], self.labels[i]

def collate_pad(batch):
    ids_list, labels = zip(*batch)
    L = max(x.shape[0] for x in ids_list)
    padded = torch.zeros(len(ids_list), L, dtype=torch.long)
    for i, x in enumerate(ids_list): padded[i, :x.shape[0]] = x
    return padded, torch.stack(labels)

class ClassificationHead(nn.Module):
    def __init__(self, hidden=4096):
        super().__init__()
        self.fc = nn.Linear(hidden, 1, bias=True)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, h): return self.fc(h.mean(dim=1)).squeeze(-1)

def find_norm(model):
    for name, mod in model.named_modules():
        if name == 'norm' or name.endswith('.norm'): return mod
    raise RuntimeError("final norm not found")

def forward_with_hook(model, ids, norm_module):
    captured = {}
    def hook(_m, _i, o): captured['h'] = o
    h = norm_module.register_forward_hook(hook)
    try: model(ids)
    finally: h.remove()
    return captured['h']

def load_weights(model, head, weights_path):
    ck = torch.load(weights_path, map_location='cpu', weights_only=True)
    md = dict(model.named_parameters())
    for k, v in ck['lora'].items():
        if k in md: md[k].data.copy_(v.to(DEVICE))
    hd = dict(head.named_parameters())
    for k, v in ck['head'].items():
        if k in hd: hd[k].data.copy_(v.to(DEVICE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights_dir', default=f'{REPO}/results/hvue_lora_weights')
    a = ap.parse_args()

    pts = sorted(glob.glob(f'{a.weights_dir}/*.pt'))
    if not pts: print("No .pt files found"); return

    results = []
    for pt in pts:
        bn = os.path.splitext(os.path.basename(pt))[0]
        parts = bn.rsplit('_', 1); lr = parts[-1]; tc = parts[0]
        for ck in ['locked_no_ft','unlocked_ft','pretrained','M_a300k']:
            if tc.endswith(ck): task = tc[:-(len(ck)+1)]; ckpt = ck; break
        else: print(f"  [skip] {bn}"); continue

        print(f"  {task} / {ckpt} / {lr} ...", flush=True)

        va_df = repro_sample(task, 'validation', N_VAL)
        n_per = N_VAL // va_df['label'].nunique()
        va_df = va_df.groupby('label', group_keys=False).apply(
            lambda g: g.sample(min(n_per, len(g)), random_state=SEED)
        ).sample(frac=1, random_state=SEED).reset_index(drop=True)

        model, tok = load_evo_model('evo-1-8k-base', DEVICE)
        maybe_load_locked_checkpoint(model, CKPTS[ckpt])
        for p in model.parameters(): p.requires_grad = False
        inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA, target_substrings=LORA_TARGETS)
        head = ClassificationHead(4096).to(DEVICE).to(AMP_DTYPE)
        load_weights(model, head, pt)

        model.eval(); head.eval()
        nm = find_norm(model)
        va_ds = HVUEDataset(va_df, tok)
        va_loader = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_pad)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for ids_v, labels_v in va_loader:
                ids_v = ids_v.to(DEVICE)
                with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                    lg_v = head(forward_with_hook(model, ids_v, nm).float())
                all_logits.append(lg_v.cpu().float().numpy())
                all_labels.append(labels_v.numpy())

        logits = np.concatenate(all_logits)
        ytrue = np.concatenate(all_labels).astype(int)
        auroc = float(roc_auc_score(ytrue, logits))
        mcc = float(matthews_corrcoef(ytrue, (logits >= 0).astype(int)))
        results.append({'task': task, 'ckpt': ckpt, 'lr': lr,
                        'auroc': round(auroc,4), 'mcc': round(mcc,4), 'n': len(ytrue)})
        print(f"    AUROC={auroc:.4f}  MCC={mcc:.4f}", flush=True)
        del model, head; torch.cuda.empty_cache()

    if results:
        print(f"\n{'Task':<22} {'Ckpt':<16} {'LR':<8} {'AUROC':>8} {'MCC':>8} {'n':>6}")
        print('-'*68)
        for r in sorted(results, key=lambda x: (x['task'], x['ckpt'])):
            print(f"{r['task']:<22} {r['ckpt']:<16} {r['lr']:<8} {r['auroc']:>8.4f} {r['mcc']:>8.4f} {r['n']:>6}")


if __name__ == '__main__':
    main()
