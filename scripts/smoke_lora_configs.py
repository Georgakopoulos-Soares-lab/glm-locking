"""Smoke test: verify LoRA layer counts and forward pass under matched/force_locked."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.utils import load_evo_model, maybe_load_locked_checkpoint
from src.lora import inject_lora, inject_lora_on_specdef

print("=== Smoke test: layer counts under each config ===")

CKPTS = [
    ("pretrained", None),
    ("M_a300k", "results/ft_locked_a300k_lr1e5_25k_locked/model_finetuned.pt"),
    ("svd_k3_a30k", "results/ft_theorem8_a30k_k3_25k_locked/model_finetuned.pt"),
]

for ckpt_name, ckpt_path in CKPTS:
    print(f"\n--- {ckpt_name} ---", flush=True)
    for cfg, targets in [
        ("full",    ["projections", "out_filter_dense", "mlp.l1", "mlp.l2", "mlp.l3"]),
        ("matched", ["out_filter_dense", "mlp.l1", "mlp.l2", "mlp.l3"]),
    ]:
        model, tok = load_evo_model('evo-1-8k-base', 'cuda')
        maybe_load_locked_checkpoint(model, ckpt_path)
        for p in model.parameters(): p.requires_grad = False
        n_wrap, _ = inject_lora(model, rank=16, alpha=32.0, target_substrings=targets)
        n_sd_avail = sum(1 for _, m in model.named_modules() if type(m).__name__ == "SpecDefLinear")
        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [{cfg}] standard_wrap={n_wrap}  SpecDefLinear available={n_sd_avail}  trainable_params={nparams:,}", flush=True)
        del model
        torch.cuda.empty_cache()

print("\n=== force_locked test on M_a300k ===", flush=True)
model, tok = load_evo_model('evo-1-8k-base', 'cuda')
maybe_load_locked_checkpoint(model, "results/ft_locked_a300k_lr1e5_25k_locked/model_finetuned.pt")
for p in model.parameters(): p.requires_grad = False
n_wrap, _ = inject_lora(model, rank=16, alpha=32.0,
                       target_substrings=["projections","out_filter_dense","mlp.l1","mlp.l2","mlp.l3"])
n_sd = inject_lora_on_specdef(model, rank=16, alpha=32.0)
nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"force_locked: standard_wrap={n_wrap} specdef_wrap={n_sd}  trainable={nparams:,}", flush=True)

ids = torch.tensor([[65, 67, 71, 84]*100], dtype=torch.long, device='cuda')
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    out = model(ids)
if isinstance(out, tuple):
    logits = out[0]
else:
    logits = out
print(f"Forward OK: logits.shape={tuple(logits.shape) if hasattr(logits, 'shape') else 'n/a'}", flush=True)
