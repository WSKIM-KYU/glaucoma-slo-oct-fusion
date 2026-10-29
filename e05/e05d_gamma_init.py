#!/usr/bin/env python
"""E05-D: GAMMA-initialized training under E05 conditions (FairVision Training/Validation only; Test never read).
Separate runner so that files used by the locked E05 evaluation (e05b_baselines.py, e05c_fusion.py) stay unchanged."""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05b_baselines as B
import e05c_fusion as C

ROOT, CACHE, log = B.ROOT, B.CACHE, B.log
RUNS_REF = B.RUNS
RUNS_D = ROOT / "results/e05/runs_d"
E03 = ROOT / "results/e03_c/runs"
SOURCE = {"oct": "octlast_concat", "concat": "octlast_concat", "cross": "octlast_cross"}
REF = {"oct": "oct_train_seed{}", "concat": "fusion_concat_imagenet_train_seed{}", "cross": "fusion_cross_imagenet_train_seed{}"}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def sub(sd, prefix):
    out = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    assert out, f"no keys with prefix {prefix}"
    return out

def build(arch, fold, seed):
    p = E03 / f"fold{fold}_seed{seed}_{SOURCE[arch]}" / "best.pt"
    assert p.is_file(), f"missing GAMMA checkpoint {p}"
    sd = torch.load(p, map_location="cpu", weights_only=True)
    if arch == "oct":
        m = B.OctNet()
        m.enc.load_state_dict(sub(sd, "oct_encoder."), strict=True)
        m.att.load_state_dict(sub(sd, "head.att."), strict=True)
        m.head[0].load_state_dict(sub(sd, "head.oct_proj.0."), strict=True)
        enc = m.enc
    else:
        m = C.FusionNet(arch)
        m.oct_enc.load_state_dict(sub(sd, "oct_encoder."), strict=True)
        m.slo_enc.load_state_dict(sub(sd, "fundus_encoder."), strict=True)
        pairs = [("oct_proj", "head.oct_proj."), ("slo_proj", "head.fundus_proj."), ("combine", "head.combine.")]
        pairs += [("att", "head.att.")] if arch == "concat" else [("cross_pos", "head.cross_pos."), ("cross_att", "head.cross_att.")]
        for dst, src in pairs:
            getattr(m, dst).load_state_dict(sub(sd, src), strict=True)
        enc = m.oct_enc
    ref = B.backbone().state_dict()
    diff = {}
    for layer in ("layer1", "layer2", "layer3", "layer4"):
        ks = [k for k in ref if k.startswith(layer) and ref[k].dtype.is_floating_point and "running" not in k]
        diff[layer] = max(float((enc.state_dict()[k] - ref[k]).abs().max()) for k in ks)
    return m, p, sha(p), diff

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
    name = f"{tag}{a.arch}_gamma_{a.bn}{'' if a.aug else '_noaug'}_seed{a.seed}"
    out = RUNS_D / name
    if (out / "metrics.json").exists() and not a.smoke:
        log(f"SKIP {name}: completed"); return
    out.mkdir(parents=True, exist_ok=True)
    model, src, src_sha, diff = build(a.arch, a.gamma_fold, a.seed)
    log(f"INIT {name} from {src.relative_to(ROOT)} sha256={src_sha[:12]}")
    log("  max |GAMMA - ImageNet| in OCT encoder: " + ", ".join(f"{k}={v:.3e}" for k, v in diff.items()))
    model = model.to(dev)
    mt = torch.tensor(B.MEAN, device=dev).view(1, 1, 3, 1, 1); st = torch.tensor(B.STD, device=dev).view(1, 1, 3, 1, 1)
    enc_prefix = ("enc.",) if a.arch == "oct" else ("oct_enc.", "slo_enc.")
    enc = [p for n, p in model.named_parameters() if n.startswith(enc_prefix)]
    head = [p for n, p in model.named_parameters() if not n.startswith(enc_prefix)]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": head, "lr": a.lr * 10}], weight_decay=1e-4)
    steps = math.ceil(len(Xtr) / a.batch); total = a.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / steps) * 0.5 * (1 + math.cos(math.pi * min(t, total) / total)))
    rng = np.random.default_rng(a.seed); gen = torch.Generator(device="cuda"); gen.manual_seed(a.seed)
    def forward(xb, aug):
        if a.arch == "oct":
            return model(B.make_input(xb, "oct", aug, gen if aug else None, dev, mt, st))
        return model(*C.make_pair(xb, aug, gen if aug else None, dev, mt, st))
    def evaluate():
        model.eval(); ps = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(Xva), a.batch):
                ps.append(torch.softmax(forward(Xva[i:i+a.batch], False).float(), 1)[:, 1].cpu())
        p = torch.cat(ps).numpy(); return p, B.metrics(yva, p)
    best, best_ep, bad, hist = -1.0, 0, 0, []
    log(f"START {name}: train={len(Xtr)} val={len(Xva)} steps/epoch={steps}")
    for ep in range(1, a.epochs + 1):
        (B.set_train if a.arch == "oct" else C.set_train)(model, a.bn)
        perm = rng.permutation(len(Xtr)); t0, tl = time.time(), 0.0
        for k in range(steps):
            idx = np.sort(perm[k * a.batch:(k + 1) * a.batch])
            yb = torch.from_numpy(ytr[idx]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(forward(Xtr[idx], a.aug), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        p, m = evaluate()
        hist.append({"epoch": ep, "train_loss": tl / steps, **m})
        log(f"{name} ep{ep:02d} loss={tl/steps:.4f} val_auroc={m['auroc']:.4f} auprc={m['auprc']:.4f} "
            f"bacc={m['balanced_acc_0p5']:.4f} mean_p={m['mean_p']:.3f} ({time.time()-t0:.0f}s)")
        if m["auroc"] > best + 1e-4:
            best, best_ep, bad = m["auroc"], ep, 0
            torch.save(model.state_dict(), out / "best.pt"); np.save(out / "val_pred_best.npy", p)
        else:
            bad += 1
            if bad >= a.patience: log(f"early stop at ep{ep}"); break
    res = {"run": name, "config": {**vars(a), "init": "gamma"}, "gamma_source": str(src.relative_to(ROOT)),
           "gamma_source_sha256": src_sha, "oct_encoder_max_abs_diff_vs_imagenet": diff,
           "best_epoch": best_ep, "best_val": next(h for h in hist if h["epoch"] == best_ep), "history": hist,
           "test_used": False, "selection": "max validation AUROC (optimistic for validation)"}
    (out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"DONE {name}: best ep{best_ep} val_auroc={best:.4f}")

def compare(n_boot=2000):
    y = pd.read_csv(CACHE / "validation_meta.csv").label.to_numpy(int)
    def load(dirpath, pattern):
        ps = [dirpath / pattern.format(s) / "val_pred_best.npy" for s in (42, 43, 44)]
        return np.mean([np.load(p) for p in ps], 0) if all(p.exists() for p in ps) else None
    rng = np.random.default_rng(20261030)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    print("=== E05-D GAMMA vs ImageNet initialization (Validation, 3-seed ensembles, exploratory) ===")
    print(f"  {'arch':8s} {'ImageNet':>9s} {'GAMMA':>9s} {'delta':>8s}  95% CI            single-seed AUROC (ImageNet | GAMMA)")
    for arch in ("oct", "concat", "cross"):
        ref = load(RUNS_REF, REF[arch]); gam = load(RUNS_D, f"{arch}_gamma_train_seed{{}}")
        if ref is None or gam is None:
            print(f"  {arch:8s} missing runs"); continue
        d = []
        for _ in range(n_boot):
            i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            d.append(roc_auc_score(y[i], gam[i]) - roc_auc_score(y[i], ref[i]))
        a_ref, a_gam = roc_auc_score(y, ref), roc_auc_score(y, gam)
        lo, hi = np.percentile(d, [2.5, 97.5])
        singles = []
        for dirpath, pat in ((RUNS_REF, REF[arch]), (RUNS_D, f"{arch}_gamma_train_seed{{}}")):
            v = [json.loads((dirpath / pat.format(s) / "metrics.json").read_text())["best_val"]["auroc"] for s in (42, 43, 44)]
            singles.append(f"{np.mean(v):.4f}±{np.std(v, ddof=1):.4f}")
        print(f"  {arch:8s} {a_ref:9.4f} {a_gam:9.4f} {a_gam - a_ref:+8.4f}  [{lo:+.4f}, {hi:+.4f}]  {singles[0]} | {singles[1]}")
    print("E05-D COMPARE DONE")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "train", "compare"])
    ap.add_argument("--arch", choices=["oct", "concat", "cross"], default="oct")
    ap.add_argument("--gamma-fold", type=int, default=1)
    ap.add_argument("--bn", choices=["train", "frozen"], default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--no-aug", dest="aug", action="store_false")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.stage == "check":
        for arch in ("oct", "concat", "cross"):
            for s in (42, 43, 44):
                _, p, h, diff = build(arch, a.gamma_fold, s)
                print(f"CHECK {arch:6s} seed{s} {p.relative_to(ROOT)} sha={h[:12]} | " + " ".join(f"{k}:{v:.2e}" for k, v in diff.items()))
        print("E05-D CHECK PASS")
    elif a.stage == "train":
        train(a)
    else:
        compare()
