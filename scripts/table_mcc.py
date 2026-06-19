"""Build full AUROC + MCC table from LoRA results."""
import csv

rows = []
with open('results/hvue_lora_all.csv') as f:
    for r in csv.DictReader(f):
        rows.append(r)

best = {}
for r in rows:
    key = (r['task'], r['ckpt'])
    auroc = float(r['best_val_auroc'])
    if key not in best or auroc > best[key][0]:
        best[key] = (auroc, r['lr'], r.get('n_val','?'))

mcc = {  # from saved preds (threshold=0.5)
    ('Host_Tropism','pretrained','1e-04'): 0.6831,
    ('Host_Tropism','unlocked_ft','1e-04'): 0.7415,
    ('Pathogenecity','M_a300k','1e-05'): 0.4952,  # NOT best LR for this combo
    ('Transmissibility','M_a300k','1e-04'): 0.7470,
    ('Transmissibility','M_a300k','5e-05'): 0.7392,
    ('Transmissibility','locked_no_ft','5e-05'): 0.7845,
    ('Transmissibility','locked_no_ft','1e-05'): 0.6725,
}
kmer = {'Host_Tropism':0.9030,'Pathogenecity':0.8459,'Transmissibility':0.9137}
cks = ['pretrained','locked_no_ft','unlocked_ft','M_a300k']
tasks = ['Host_Tropism','Pathogenecity','Transmissibility']

# === Full table: best LR per combo ===
print("=== BEST-LR PER COMBO: AUROC + MCC (where saved preds exist) ===\n")
header = f"{'Task':<22} {'k-mer':>8} |"
for c in cks:
    header += f" {c:<22} |"
print(header)
sep = "-" * len(header)
print(sep)
for t in tasks:
    line = f"{t:<22} {kmer[t]:>8.4f} |"
    for c in cks:
        key = (t, c)
        if key in best:
            auroc, lr, _ = best[key]
            m = mcc.get((t, c, lr))
            if m is not None:
                cell = f"{auroc:.4f} MCC={m:.4f}"
            else:
                cell = f"{auroc:.4f} MCC=?"
            line += f" {cell:<22} |"
        else:
            line += f" {'--':<22} |"
    print(line)

# === All LR combos with saved MCC ===
print(f"\n=== ALL SAVED PRED COMBO: AUROC + MCC (n=2000 val, thr=0.5) ===\n")
print(f"{'Task':<22} {'Ckpt':<16} {'LR':<8} {'AUROC':>8} {'MCC':>8}")
print("-" * 66)
for (t, c, lr), mcc_val in sorted(mcc.items()):
    auroc_val = None
    for r in rows:
        if r['task']==t and r['ckpt']==c and r['lr']==lr:
            auroc_val = float(r['best_val_auroc'])
            break
    a_str = f"{auroc_val:.4f}" if auroc_val else "?"
    print(f"{t:<22} {c:<16} {lr:<8} {a_str:>8} {mcc_val:>8.4f}")
