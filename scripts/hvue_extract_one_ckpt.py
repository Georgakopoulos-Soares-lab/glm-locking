"""Extract HVUE embeddings for ONE checkpoint across all (task, split) pairs.

Loads the model once, iterates over tasks and splits, writes one npz per pair.
Designed to be launched per-GPU.
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.utils import load_evo_model, maybe_load_locked_checkpoint, get_amp_settings


@torch.no_grad()
def get_hidden(model, ids, amp_dtype, hook_target):
    captured = {}
    def hook(_m, _inp, out):
        captured['h'] = out.detach()
    h = hook_target.register_forward_hook(hook)
    try:
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            _ = model(ids)
    finally:
        h.remove()
    return captured['h'].float().mean(dim=1)  # [B, D]


def find_norm(model):
    for name, mod in model.named_modules():
        if name == 'norm' or name.endswith('.norm'):
            return mod
    raise RuntimeError("no final norm")


def process_split(model, tok, hook_target, amp_dtype, ckpt_name, task, split, n, batch, max_len, device, out_dir):
    out = f'{out_dir}/{ckpt_name}_{task}_{split}.npz'
    if os.path.exists(out):
        print(f'  skip exists: {out}')
        return
    df = pd.read_parquet(f'data/hvue/{task}_{split}.parquet')
    if n and len(df) > n:
        per_class = n // df['label'].nunique()
        df = df.groupby('label', group_keys=False).apply(
            lambda g: g.sample(min(per_class, len(g)), random_state=0)
        )
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    print(f'  [{task}/{split}] n={len(df)}, label_mean={df.label.mean():.3f}')
    feats = []; labels = []
    t0 = time.time()
    for i in range(0, len(df), batch):
        b = df.iloc[i:i+batch]
        seqs = [s[:max_len] for s in b['sequence'].tolist()]
        ids_list = [tok.tokenize(s) for s in seqs]
        L = max(len(x) for x in ids_list)
        ids = torch.zeros((len(ids_list), L), dtype=torch.long)
        for j, x in enumerate(ids_list):
            ids[j, :len(x)] = torch.tensor(x)
        ids = ids.to(device)
        try:
            f = get_hidden(model, ids, amp_dtype, hook_target).cpu().numpy()
            feats.append(f); labels.extend(b['label'].tolist())
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for k in range(len(seqs)):
                f = get_hidden(model, ids[k:k+1], amp_dtype, hook_target).cpu().numpy()
                feats.append(f); labels.append(b['label'].iloc[k])
        if (i // batch) % 100 == 0 and i > 0:
            rate = (i+batch) / (time.time()-t0)
            print(f'    {i+batch}/{len(df)}  ({rate:.1f} seq/s)')
    X = np.concatenate(feats, axis=0); y = np.array(labels)
    np.savez_compressed(out, X=X, y=y, ckpt=ckpt_name, task=task, split=split)
    print(f'  wrote {out}  X={X.shape}  in {time.time()-t0:.1f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_name', required=True)
    ap.add_argument('--ckpt_path', default="")
    ap.add_argument('--n_train', type=int, default=3000)
    ap.add_argument('--n_val', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out_dir', default='results/hvue_embeddings')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    amp_dtype, _ = get_amp_settings()
    print(f'Loading model for {a.ckpt_name}...')
    model, tok = load_evo_model('evo-1-8k-base', a.device)
    if a.ckpt_path:
        maybe_load_locked_checkpoint(model, a.ckpt_path)
    model.eval()
    hook_target = find_norm(model)

    for task in ['Host_Tropism', 'Pathogenecity', 'Transmissibility']:
        for split, n in [('train', a.n_train), ('validation', a.n_val)]:
            try:
                process_split(model, tok, hook_target, amp_dtype, a.ckpt_name, task, split, n, a.batch, a.max_len, a.device, a.out_dir)
            except Exception as e:
                print(f'  ERR [{task}/{split}]: {e}')
                import traceback; traceback.print_exc()
    print('DONE')


if __name__ == '__main__':
    main()
