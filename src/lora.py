"""Minimal LoRA adapter for nn.Linear, no peft dependency.

Wraps a frozen Linear with a low-rank delta:  y = W·x + (α/r) · B·A·x
A: (r, in), B: (out, r), both trainable. Init: A ~ kaiming, B = 0.
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        in_f, out_f = base.in_features, base.out_features
        # use bf16/fp16-friendly dtype matching base
        dtype = base.weight.dtype
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # B stays zero -> initial output equals base output

    def forward(self, x):
        out = self.base(x)
        # LoRA delta in same dtype as input
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scaling * delta


def inject_lora(model: nn.Module, rank: int, alpha: float, target_substrings):
    """Replace every nn.Linear whose qualified name contains any substring with LoRALinear.

    Returns (n_wrapped, list_of_lora_param_names).
    """
    # Skip any module nested inside a SpecDefLinear (its `.comp` and `.linear`
    # are referenced by lock_specdef.forward via `.weight` and must stay raw nn.Linear).
    skip_prefixes = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "SpecDefLinear":
            skip_prefixes.append(name + ".")

    to_replace = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not any(s in name for s in target_substrings):
            continue
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        to_replace.append(name)

    n = 0
    for full_name in to_replace:
        parent = model
        parts = full_name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]
        base = getattr(parent, leaf)
        wrapped = LoRALinear(base, rank=rank, alpha=alpha)
        setattr(parent, leaf, wrapped)
        n += 1

    lora_param_names = []
    for name, p in model.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            lora_param_names.append(name)
    return n, lora_param_names
