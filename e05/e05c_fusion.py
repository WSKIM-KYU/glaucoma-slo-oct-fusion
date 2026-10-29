#!/usr/bin/env python
"""E05-C: OCT+SLO fusion (Concat, Cross-Attention) on FairVision Training/Validation. Test never read."""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05b_baselines as B

RUNS, CACHE, log = B.RUNS, B.CACHE, B.log

def jitter(t, gen, dev):
    b = t.shape[0]
    c = 1 + (torch.rand(b, 1, 1, 1, device=dev, generator=gen) - .5) * .2
    sh = (torch.rand(b, 1, 1, 1, device=dev, generator=gen) - .5) * .1
    mu = t.mean(dim=(1, 2, 3), keepdim=True)
    return ((t - mu) * c + mu + sh).clamp_(0, 1)

def make_pair(xb, aug, gen, dev, mt, st):
    x = torch.from_numpy(xb).to(dev, non_blocking=True).float().div_(255.0)
    if aug:
        flip = torch.rand(x.shape[0], device=dev, generator=gen) < 0.5
        x[flip] = x[flip].flip(-1)
        x = torch.cat([jitter(x[:, :1], gen, dev), jitter(x[:, 1:], gen, dev)], 1)
    x = (x.unsqueeze(2).expand(-1, -1, 3, -1, -1) - mt) / st
    return x[:, 0], x[:, 1:]

class FusionNet(nn.Module):
    def __init__(s, mode):
        super().__init__(); s.mode = mode
        s.oct_enc, s.slo_enc = B.backbone(), B.backbone()
        s.att = nn.Sequential(nn.Linear(513, 96), nn.Tanh(), nn.Linear(96, 1))
        s.oct_proj = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.25))
        s.slo_proj = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.25))
        s.cross_pos = nn.Linear(1, 128)
        s.cross_att = nn.MultiheadAttention(128, 4, batch_first=True)
        s.combine = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.25))
        s.cls = nn.Linear(128, 2)
        s.register_buffer("pos", torch.linspace(-1, 1, B.N_SLICES).view(1, B.N_SLICES, 1))
    def forward(s, slo, octs):
        b, n = octs.shape[:2]
        zo = s.oct_enc(octs.flatten(0, 1)).view(b, n, 512)
        f = s.slo_proj(s.slo_enc(slo))
        pos = s.pos.to(zo.dtype).expand(b, -1, -1)
        if s.mode == "cross":
            tok = s.oct_proj(zo) + s.cross_pos(pos)
            ctx, _ = s.cross_att(f[:, None, :], tok, tok, need_weights=False)
            z = s.combine(torch.cat([f, ctx[:, 0]], -1))
        else:
            w = torch.softmax(s.att(torch.cat([zo, pos], -1)).squeeze(-1), 1)
            o = s.oct_proj((w.unsqueeze(-1) * zo).sum(1))
            z = s.combine(torch.cat([o, f], -1))
        return s.cls(z)

def load_unimodal(model, seed):
    for enc, kind in ((model.oct_enc, "oct"), (model.slo_enc, "slo")):
        sd = torch.load(RUNS / f"{kind}_train_seed{seed}" / "best.pt", map_location="cpu", weights_only=True)
        enc.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("enc.")}, strict=True)

def set_train(model, bn):
    model.train()
    if bn == "frozen":
        for enc in (model.oct_enc, model.slo_enc):
            for m in enc.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()

def evaluate(model, X, y, bs, dev, mt, st):
    model.eval(); ps = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), bs):
            slo, octs = make_pair(X[i:i+bs], False, None, dev, mt, st)
            ps.append(torch.softmax(model(slo, octs).float(), 1)[:, 1].cpu())
    p = torch.cat(ps).numpy()
    return p, B.metrics(y, p)

def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda"); assert torch.cuda.is_available(), "CUDA required"
    Xtr = np.load(CACHE / "training_x.npy"); ytr = pd.read_csv(CACHE / "training_meta.csv").label.to_numpy(int)
    Xva = np.load(CACHE / "validation_x.npy"); yva = pd.read_csv(CACHE / "validation_meta.csv").label.to_numpy(int)
    assert len(Xtr) == 6000 and len(Xva) == 1000
    tag = "smoke_" if a.smoke else ""
    if a.smoke:
        r = np.random.default_rng(0)
        it, iv = np.sort(r.choice(6000, 256, replace=False)), np.sort(r.choice(1000, 200, replace=False))
        Xtr, ytr, Xva, yva = Xtr[it], ytr[it], Xva[iv], yva[iv]
    name = f"{tag}fusion_{a.mode}_{a.init}_{a.bn}{'' if a.aug else '_noaug'}_seed{a.seed}"
    out = RUNS / name
    if (out / "metrics.json").exists() and not a.smoke:
        log(f"SKIP {name}: completed"); return
    out.mkdir(parents=True, exist_ok=True)
    model = FusionNet(a.mode)
    if a.init == "unimodal": load_unimodal(model, a.seed)
    model = model.to(dev)
    mt = torch.tensor(B.MEAN, device=dev).view(1, 1, 3, 1, 1); st = torch.tensor(B.STD, device=dev).view(1, 1, 3, 1, 1)
    enc = list(model.oct_enc.parameters()) + list(model.slo_enc.parameters())
    head = [p for n, p in model.named_parameters() if not n.startswith(("oct_enc.", "slo_enc."))]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": head, "lr": a.lr * 10}], weight_decay=1e-4)
    steps = math.ceil(len(Xtr) / a.batch); total = a.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / steps) * 0.5 * (1 + math.cos(math.pi * min(t, total) / total)))
    rng = np.random.default_rng(a.seed); gen = torch.Generator(device="cuda"); gen.manual_seed(a.seed)
    best, best_ep, bad, hist = -1.0, 0, 0, []
    log(f"START {name}: train={len(Xtr)} val={len(Xva)} steps/epoch={steps}")
    for ep in range(1, a.epochs + 1):
        set_train(model, a.bn); perm = rng.permutation(len(Xtr)); t0, tl = time.time(), 0.0
        for k in range(steps):
            idx = np.sort(perm[k * a.batch:(k + 1) * a.batch])
            yb = torch.from_numpy(ytr[idx]).to(dev)
            slo, octs = make_pair(Xtr[idx], a.aug, gen, dev, mt, st)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(slo, octs), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        p, m = evaluate(model, Xva, yva, a.batch, dev, mt, st)
        hist.append({"epoch": ep, "train_loss": tl / steps, **m})
        log(f"{name} ep{ep:02d} loss={tl/steps:.4f} val_auroc={m['auroc']:.4f} auprc={m['auprc']:.4f} "
            f"bacc={m['balanced_acc_0p5']:.4f} mean_p={m['mean_p']:.3f} ({time.time()-t0:.0f}s)")
        if m["auroc"] > best + 1e-4:
            best, best_ep, bad = m["auroc"], ep, 0
            torch.save(model.state_dict(), out / "best.pt"); np.save(out / "val_pred_best.npy", p)
        else:
            bad += 1
            if bad >= a.patience: log(f"early stop at ep{ep}"); break
    res = {"run": name, "config": vars(a), "best_epoch": best_ep,
           "best_val": next(h for h in hist if h["epoch"] == best_ep), "history": hist,
           "test_used": False, "selection": "max validation AUROC (optimistic for validation)"}
    (out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"DONE {name}: best ep{best_ep} val_auroc={best:.4f}")

def compare(n_boot=2000):
    y = pd.read_csv(CACHE / "validation_meta.csv").label.to_numpy(int)
    groups = {}
    for f in sorted(RUNS.glob("*/metrics.json")):
        r = json.loads(f.read_text()); c = r["config"]
        if c.get("smoke") or not (f.parent / "val_pred_best.npy").exists(): continue
        key = f"{c['model']}_{c['bn']}" if "model" in c else f"fusion_{c['mode']}_{c['init']}_{c['bn']}"
        if not c.get("aug", True): key += "_noaug"
        groups.setdefault(key, {})[c["seed"]] = np.load(f.parent / "val_pred_best.npy")
    print("=== E05-C seed-ensemble comparison (Validation, exploratory) ===")
    ens = {k: np.mean([v[s] for s in sorted(v)], 0) for k, v in groups.items() if len(v) == 3}
    for k, p in ens.items():
        print(f"{k:32s} seeds=3 ensemble_auroc={roc_auc_score(y, p):.4f}")
    ref = "oct_train"
    if ref not in ens: print("reference oct_train ensemble missing"); return
    rng = np.random.default_rng(20261028)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    print(f"\nPaired stratified bootstrap ({n_boot}x), delta AUROC vs {ref}:")
    for k, p in ens.items():
        if k == ref: continue
        d = []
        for _ in range(n_boot):
            i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            d.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], ens[ref][i]))
        d = np.array(d)
        print(f"  {k:32s} delta={roc_auc_score(y, p) - roc_auc_score(y, ens[ref]):+.4f} "
              f"95%CI=[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
    print("E05-C COMPARE DONE")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "compare"])
    ap.add_argument("--mode", choices=["concat", "cross"], default="concat")
    ap.add_argument("--init", choices=["imagenet", "unimodal"], default="imagenet")
    ap.add_argument("--bn", choices=["train", "frozen"], default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--no-aug", dest="aug", action="store_false")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    train(a) if a.stage == "train" else compare()
