#!/usr/bin/env python
"""E05-B: in-domain FairVision baselines (OCT-only, SLO-only). Training/Validation only; Test never read."""
import argparse, io, json, math, os, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data/external/fairvision/Glaucoma/Dataset/dataset.zip"
AUD = ROOT / "results/e05/audit"
CACHE = ROOT / "results/e05/cache"
RUNS = ROOT / "results/e05/runs"
N_SLICES, BSCAN_AXIS = 8, 0
IDX = np.rint(np.linspace(0, 199, N_SLICES)).astype(int)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def log(msg): print(time.strftime("[%H:%M:%S] ") + msg, flush=True)

def to224(a):
    if a.dtype != np.uint8 or a.shape != (200, 200):
        raise ValueError(f"unexpected {a.dtype} {a.shape}")
    return np.asarray(Image.fromarray(a).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)

def prepare():
    for split, fn in (("training", "e05_train_manifest.csv"), ("validation", "e05_val_manifest.csv")):
        m = pd.read_csv(AUD / fn, dtype={"case_id": str})
        assert set(m.split) == {split}, f"manifest split mismatch: {fn}"
        out, meta = CACHE / f"{split}_x.npy", CACHE / f"{split}_meta.csv"
        if out.exists() and meta.exists():
            log(f"SKIP {split}: cache exists"); continue
        tmp = CACHE / f"{split}_x.partial.npy"
        x = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(m), 1 + N_SLICES, 224, 224))
        with zipfile.ZipFile(ZIP) as z:
            for i, r in enumerate(m.itertuples(index=False)):
                with np.load(io.BytesIO(z.read(r.npz_path)), allow_pickle=False) as d:
                    slo, vol, lab = np.asarray(d["slo_fundus"]), np.asarray(d["oct_bscans"]), int(d["glaucoma"])
                if lab != int(r.label):
                    raise RuntimeError(f"label mismatch case {r.case_id}: npz={lab} manifest={r.label}")
                x[i, 0] = to224(slo)
                for j, k in enumerate(IDX):
                    x[i, 1 + j] = to224(np.take(vol, k, axis=BSCAN_AXIS))
                if (i + 1) % 500 == 0: log(f"prepare {split} {i+1}/{len(m)}")
        x.flush(); del x
        os.replace(tmp, out)
        m[["case_id", "label"]].to_csv(meta, index=False)
        log(f"DONE {split}: {len(m)} cases -> {out}")

def backbone():
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Identity(); return m

class OctNet(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = backbone()
        s.att = nn.Sequential(nn.Linear(513, 96), nn.Tanh(), nn.Linear(96, 1))
        s.head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.25), nn.Linear(128, 2))
        s.register_buffer("pos", torch.linspace(-1, 1, N_SLICES).view(1, N_SLICES, 1))
    def forward(s, x):
        b, n = x.shape[:2]
        z = s.enc(x.flatten(0, 1)).view(b, n, 512)
        w = torch.softmax(s.att(torch.cat([z, s.pos.to(z.dtype).expand(b, -1, -1)], -1)).squeeze(-1), 1)
        return s.head((w.unsqueeze(-1) * z).sum(1))

class SloNet(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = backbone()
        s.head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.25), nn.Linear(128, 2))
    def forward(s, x): return s.head(s.enc(x[:, 0]))

def set_train(model, bn):
    model.train()
    if bn == "frozen":
        for mod in model.enc.modules():
            if isinstance(mod, nn.BatchNorm2d): mod.eval()

def metrics(y, p):
    yh = (p >= .5).astype(int)
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "balanced_acc_0p5": float(balanced_accuracy_score(y, yh)),
            "sens_0p5": float(((yh == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
            "spec_0p5": float(((yh == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
            "brier": float(np.mean((p - y) ** 2)), "mean_p": float(p.mean())}

def make_input(xb, kind, aug, gen, dev, mt, st):
    x = torch.from_numpy(xb).to(dev, non_blocking=True)
    x = (x[:, 1:] if kind == "oct" else x[:, :1]).float().div_(255.0)
    if aug:
        b = x.shape[0]
        flip = torch.rand(b, device=dev, generator=gen) < 0.5
        x[flip] = x[flip].flip(-1)
        c = 1 + (torch.rand(b, 1, 1, 1, device=dev, generator=gen) - .5) * .2
        sh = (torch.rand(b, 1, 1, 1, device=dev, generator=gen) - .5) * .1
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - mu) * c + mu + sh).clamp_(0, 1)
    x = x.unsqueeze(2).expand(-1, -1, 3, -1, -1)
    return (x - mt) / st

def evaluate(model, X, y, kind, bs, dev, mt, st):
    model.eval(); ps = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), bs):
            ps.append(torch.softmax(model(make_input(X[i:i+bs], kind, False, None, dev, mt, st)).float(), 1)[:, 1].cpu())
    p = torch.cat(ps).numpy()
    return p, metrics(y, p)

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
    name = f"{tag}{a.model}_{a.bn}{'' if a.aug else '_noaug'}_seed{a.seed}"
    out = RUNS / name
    if (out / "metrics.json").exists() and not a.smoke:
        log(f"SKIP {name}: completed"); return
    out.mkdir(parents=True, exist_ok=True)
    model = (OctNet() if a.model == "oct" else SloNet()).to(dev)
    mt = torch.tensor(MEAN, device=dev).view(1, 1, 3, 1, 1); st = torch.tensor(STD, device=dev).view(1, 1, 3, 1, 1)
    enc = list(model.enc.parameters()); head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(make_input(Xtr[idx], a.model, a.aug, gen, dev, mt, st)), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        p, m = evaluate(model, Xva, yva, a.model, a.batch, dev, mt, st)
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

def summary():
    rows = []
    for f in sorted(RUNS.glob("*/metrics.json")):
        r = json.loads(f.read_text()); c = r["config"]
        if c.get("smoke"): continue
        rows.append({"model": c["model"], "bn": c["bn"], "aug": c["aug"], "seed": c["seed"],
                     "best_epoch": r["best_epoch"], **{k: r["best_val"][k] for k in ("auroc", "auprc", "balanced_acc_0p5", "brier", "mean_p")}})
    if not rows: print("no completed runs"); return
    df = pd.DataFrame(rows).sort_values(["model", "bn", "aug", "seed"])
    print("=== E05-B per-run (Validation, best epoch) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    g = df.groupby(["model", "bn", "aug"])[["auroc", "auprc", "balanced_acc_0p5", "brier"]].agg(["mean", "std"])
    print("\n=== E05-B mean ± sd over seeds ===")
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))
    df.to_csv(RUNS.parent / "e05b_summary.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prepare", "train", "summary"])
    ap.add_argument("--model", choices=["oct", "slo"], default="oct")
    ap.add_argument("--bn", choices=["train", "frozen"], default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--no-aug", dest="aug", action="store_false")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "train": lambda: train(a), "summary": summary}[a.stage]()
