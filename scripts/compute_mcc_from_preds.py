"""Compute MCC from saved best-val predictions (threshold=0.5)."""
import numpy as np, os, glob
from sklearn.metrics import matthews_corrcoef

pred_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results', 'hvue_lora_preds')
files = sorted(glob.glob(f'{pred_dir}/*.npz'))
print(f'{len(files)} pred files found\n')
print(f'{"task":<20} {"ckpt":<16} {"lr":<8} {"MCC":>8}  n_val')
print('-' * 65)

for f in files:
    data = np.load(f)
    preds = data['preds']
    labels = data['labels']
    pred_bin = (preds >= 0.5).astype(int)
    mcc = matthews_corrcoef(labels, pred_bin)
    basename = os.path.splitext(os.path.basename(f))[0]
    # Parse: {task}_{ckpt}_{lr}
    parts = basename.rsplit('_', 1)
    lr = parts[-1]
    task_ckpt = parts[0]
    for ck in ['locked_no_ft', 'unlocked_ft', 'pretrained', 'M_a300k']:
        if task_ckpt.endswith(ck):
            task = task_ckpt[:-(len(ck)+1)]
            ckpt = ck
            break
    else:
        task, ckpt = task_ckpt, '?'
    print(f'{task:<20} {ckpt:<16} {lr:<8} {mcc:>8.4f}  n={len(labels)}')
