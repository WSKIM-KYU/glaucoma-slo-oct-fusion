#!/usr/bin/env python
"""Revision 1: concatenation fusion with on-the-fly gradient modulation (OGM; Peng et al., CVPR 2022).
FairVision Training/Validation only; the Test split is never read. Run from the repository root:

    python -u ~/r1/r1_ogm_concat.py --stage train --seed 42 --smoke     # 1-2 min sanity check
    for s in 42 43 44; do python -u ~/r1/r1_ogm_concat.py --stage train --seed $s; done
    python -u ~/r1/r1_ogm_concat.py --stage compare

Everything except the modulation is identical to e05/e05c_fusion.py (--mode concat --init imagenet --bn train):
same model, data, augmentation, optimizer, schedule, epochs, patience, seeds and checkpoint selection.

Modulation. At every step the contribution of each modality is scored as the mean probability assigned
to the true class when the other modality's projected feature is set to zero before the combination layer
(no gradient, no dropout). With ratio rho = score(dominant) / score(other) > 1, the update of the dominant
modality's ENCODER is scaled by 1 - tanh(alpha * (rho - 1)); the other encoder and all fusion layers are
unchanged. Because AdamW normalizes gradient magnitude, scaling the gradient itself would have almost no
effect, so the scaling is applied to that encoder's step size (equivalent to gradient scaling under SGD,
which the original method used). The Gaussian-noise enhancement (GE) of the original is not used.

Outputs go to ~/r1/runs_ogm (the repository is not modified).
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "e05"))
import e05b_baselines as B
import e05c_fusion as C

CACHE, log = B.CACHE, B.log
OUT = Path.home() / "r1" / "runs_ogm"
SEEDS = (42, 43, 44)


def forward_parts(m, slo, octs):
    """Same computation as FusionNet.forward for mode='concat'; also returns the two projected features."""
    b, n = octs.shape[:2]
    zo = m.oct_enc(octs.flatten(0, 1)).view(b, n, 512)
    f = m.slo_proj(m.slo_enc(slo))
    pos = m.pos.to(zo.dtype).expand(b, -1, -1)
    w = torch.softmax(m.att(torch.cat([zo, pos], -1)).squeeze(-1), 1)
    o = m.oct_proj((w.unsqueeze(-1) * zo).sum(1))
    return m.cls(m.combine(torch.cat([o, f], -1))), o, f


@torch.no_grad()
def modality_scores(m, o, f, y):
    def score(oo, ff):
        z = F.relu(m.combine[0](torch.cat([oo, ff], -1)))  # combination layer without dropout
        return torch.softmax(m.cls(z).float(), 1).gather(1, y[:, None]).sum()
    zero_o, zero_f = torch.zeros_like(o), torch.zeros_like(f)
    return score(o, zero_f), score(zero_o, f)  # OCT-only path, SLO-only path


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
    name = f"{tag}fusion_concat_imagenet_train_ogm_a{a.alpha:g}_seed{a.seed}"
    out = OUT / name
    if (out / "metrics.json").exists() and not a.smoke:
        log(f"SKIP {name}: completed"); return
    out.mkdir(parents=True, exist_ok=True)
    model = C.FusionNet("concat").to(dev)
    mt = torch.tensor(B.MEAN, device=dev).view(1, 1, 3, 1, 1); st = torch.tensor(B.STD, device=dev).view(1, 1, 3, 1, 1)
    head = [p for n, p in model.named_parameters() if not n.startswith(("oct_enc.", "slo_enc."))]
    # Same learning rates and weight decay as e05c_fusion.py; the two encoders are separate groups so that
    # their step sizes can be modulated independently.
    opt = torch.optim.AdamW([{"params": list(model.oct_enc.parameters()), "lr": a.lr},
                             {"params": list(model.slo_enc.parameters()), "lr": a.lr},
                             {"params": head, "lr": a.lr * 10}], weight_decay=1e-4)
    steps = math.ceil(len(Xtr) / a.batch); total = (1 if a.smoke else a.epochs) * steps
    epochs = 1 if a.smoke else a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / steps) * 0.5 * (1 + math.cos(math.pi * min(t, total) / total)))
    rng = np.random.default_rng(a.seed); gen = torch.Generator(device="cuda"); gen.manual_seed(a.seed)
    best, best_ep, bad, hist = -1.0, 0, 0, []
    log(f"START {name}: train={len(Xtr)} val={len(Xva)} steps/epoch={steps} alpha={a.alpha}")
    for ep in range(1, epochs + 1):
        C.set_train(model, "train"); perm = rng.permutation(len(Xtr)); t0, tl = time.time(), 0.0
        rhos, co, cs = [], [], []
        for k in range(steps):
            idx = np.sort(perm[k * a.batch:(k + 1) * a.batch])
            yb = torch.from_numpy(ytr[idx]).to(dev)
            slo, octs = C.make_pair(Xtr[idx], a.aug, gen, dev, mt, st)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, o, f = forward_parts(model, slo, octs)
                loss = F.cross_entropy(logits, yb)
                s_oct, s_slo = modality_scores(model, o.detach(), f.detach(), yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            rho = float(s_oct / s_slo.clamp_min(1e-8))  # > 1: OCT dominates
            c_oct = 1 - math.tanh(a.alpha * (rho - 1)) if rho > 1 else 1.0
            c_slo = 1 - math.tanh(a.alpha * (1 / rho - 1)) if rho < 1 else 1.0
            opt.param_groups[0]["lr"] *= c_oct; opt.param_groups[1]["lr"] *= c_slo
            opt.step(); sched.step()  # LambdaLR recomputes all learning rates from the base values
            tl += loss.item(); rhos.append(rho); co.append(c_oct); cs.append(c_slo)
        p, m = C.evaluate(model, Xva, yva, a.batch, dev, mt, st)
        diag = {"rho_oct_over_slo_median": float(np.median(rhos)), "frac_steps_oct_dominant": float(np.mean(np.array(rhos) > 1)),
                "coef_oct_mean": float(np.mean(co)), "coef_slo_mean": float(np.mean(cs))}
        hist.append({"epoch": ep, "train_loss": tl / steps, **m, **diag})
        log(f"{name} ep{ep:02d} loss={tl/steps:.4f} val_auroc={m['auroc']:.4f} mean_p={m['mean_p']:.3f} | "
            f"rho(OCT/SLO) median={diag['rho_oct_over_slo_median']:.3f} OCT-dominant steps={diag['frac_steps_oct_dominant']:.2f} "
            f"coef_oct={diag['coef_oct_mean']:.3f} coef_slo={diag['coef_slo_mean']:.3f} ({time.time()-t0:.0f}s)")
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


def compare(alpha, n_boot=2000):
    y = pd.read_csv(CACHE / "validation_meta.csv").label.to_numpy(int)
    ens = lambda d, pat: np.mean([np.load(d / pat.format(s) / "val_pred_best.npy") for s in SEEDS], 0)
    P = {"OCT-only": ens(B.RUNS, "oct_train_seed{}"),
         "concat (standard)": ens(B.RUNS, "fusion_concat_imagenet_train_seed{}"),
         f"concat + OGM (alpha={alpha:g})": ens(OUT, "fusion_concat_imagenet_train_ogm_a%g_seed{}" % alpha)}
    print("=== R1: MODALITY-BALANCED CONCATENATION FUSION (validation, 3-seed ensembles) ===")
    for s in SEEDS:
        r = json.loads((OUT / f"fusion_concat_imagenet_train_ogm_a{alpha:g}_seed{s}" / "metrics.json").read_text())
        b = r["best_val"]
        print(f"  seed {s}: best epoch {r['best_epoch']} val_auroc={b['auroc']:.4f} | rho median={b['rho_oct_over_slo_median']:.3f} "
              f"OCT-dominant steps={b['frac_steps_oct_dominant']:.2f} coef_oct={b['coef_oct_mean']:.3f} coef_slo={b['coef_slo_mean']:.3f}")
    for k, p in P.items(): print(f"{k:32s} ensemble AUROC = {roc_auc_score(y, p):.4f}")
    rng = np.random.default_rng(20261119); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    idx = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n_boot)]
    new = list(P)[2]
    for ref in ("OCT-only", "concat (standard)"):
        d = [roc_auc_score(y[i], P[new][i]) - roc_auc_score(y[i], P[ref][i]) for i in idx]
        print(f"delta AUROC, {new} minus {ref}: {roc_auc_score(y, P[new]) - roc_auc_score(y, P[ref]):+.4f} "
              f"95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
    print("R1 OGM COMPARE DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "compare"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--no-aug", dest="aug", action="store_false")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    train(a) if a.stage == "train" else compare(a.alpha)
