"""Relative change of OCT encoder layer 4 between GAMMA-trained and ImageNet weights. Run from repository root."""
import torch, numpy as np
from torchvision.models import resnet18, ResNet18_Weights
ref = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
print("=== relative change of OCT encoder layer4 (GAMMA vs ImageNet) ===")
print("checkpoint                         ||dW||/||W||   max|dW|/max|W|   changed params (|dW|>1e-6)")
for conf in ("octlast_concat", "octlast_cross"):
    for s in (42, 43, 44):
        sd = torch.load(f"results/e03_c/runs/fold1_seed{s}_{conf}/best.pt", map_location="cpu", weights_only=True)
        dn = wn = 0.0; dmax = wmax = 0.0; ch = tot = 0
        for k, w in ref.items():
            if not k.startswith("layer4") or not w.dtype.is_floating_point or "running" in k: continue
            d = sd["oct_encoder." + k].float() - w.float()
            dn += float((d ** 2).sum()); wn += float((w.float() ** 2).sum())
            dmax = max(dmax, float(d.abs().max())); wmax = max(wmax, float(w.abs().max()))
            ch += int((d.abs() > 1e-6).sum()); tot += d.numel()
        print(f"{conf}_seed{s:<6d} {np.sqrt(dn / wn):14.2e} {dmax / wmax:16.2e} {100 * ch / tot:12.1f}%")
print("DONE")
