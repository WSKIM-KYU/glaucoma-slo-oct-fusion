#!/usr/bin/env python
"""Revision 1 checks for diagnostics-4696139. Run from the repository root:

    python -u analysis/r1_revision_checks.py --stage dates       # test-evaluation date
    python -u analysis/r1_revision_checks.py --stage md          # minimum MD of normal-labelled test eyes
    python -u analysis/r1_revision_checks.py --stage s6          # Table S6: probability vs logit fusion (validation)
    python -u analysis/r1_revision_checks.py --stage gamma_oof   # GAMMA out-of-fold binary AUROC / sensitivity
    python -u analysis/r1_revision_checks.py --stage slo_ablation  # Table S7 (validation only, needs GPU)

No stage reads test images or refits anything used by the locked evaluation. `md` reads locked test
predictions and metadata only (post hoc, descriptive). Nothing is written except printed output.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "e05"))
SEEDS = (42, 43, 44)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


def dates():
    ft = ROOT / "results/e05/formal_test"
    fmt = lambda p: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
    print("protocol.json 'created':", json.loads((ft / "protocol.json").read_text())["created"])
    for p in [ft / "protocol.json", ft / "results.json", ft / "supplementary_stats.json"]:
        print(f"{p.name:28s} mtime {fmt(p)}" if p.exists() else f"{p.name} missing")
    preds = sorted((ft / "test_preds").glob("*.npy"))
    if preds:
        print("test_preds: first", fmt(min(preds, key=lambda q: q.stat().st_mtime)),
              "| last", fmt(max(preds, key=lambda q: q.stat().st_mtime)))
    print("-> report the results.json mtime as the date of the single test evaluation "
          "(check that the file was not copied or touched afterwards).")


def md():
    import e05_formal_test as T
    cal = json.loads(T.PROTOCOL.read_text())["calibration"]["oct"]
    meta = pd.read_csv(T.CACHE / "test_meta.csv", dtype={"case_id": str}); y = meta.label.to_numpy(int)
    Z = T.cand_logits(T.ens(lambda n: np.load(T.PRED / f"{n}.npy")))
    p = T.sig(cal["platt_a"] * Z["oct"] + cal["platt_b"]); thr = cal["thr_spec85_val"]
    sm = pd.read_csv(T.SUMMARY_CSV, dtype={"filename": str})
    sm["case_id"] = sm.filename.str.extract(r"(\d{5})\.npz$")[0]; sm = sm.set_index("case_id").loc[meta.case_id]
    mdv = pd.to_numeric(sm.md, errors="coerce").to_numpy()
    race = sm.race.astype(str).str.strip().str.lower().to_numpy()
    n = y == 0
    print("=== NORMAL-LABELLED TEST EYES: MD ===")
    print(f"n normal = {n.sum()}, MD available = {np.isfinite(mdv[n]).sum()}")
    print(f"minimum MD = {np.nanmin(mdv[n]):.2f} dB | n with MD < -1 dB = {(mdv[n] < -1).sum()} | n with MD < -2 dB = {(mdv[n] < -2).sum()}")
    g = y == 1
    print(f"glaucoma eyes: maximum MD = {np.nanmax(mdv[g]):.2f} dB | n with MD > -3 dB = {(mdv[g] > -3).sum()}")
    print("false positives at locked thr85 by race (should be white 162/1238, black 51/163, asian 4/133):")
    for r in ("white", "black", "asian"):
        m = n & (race == r); print(f"  {r:6s} {(p[m] >= thr).sum()}/{m.sum()}")


def s6():
    R = ROOT / "results/e05/runs"
    y = pd.read_csv(ROOT / "results/e05/cache/validation_meta.csv").label.to_numpy(int)
    ens = lambda n: np.mean([np.load(R / f"{n}_seed{s}" / "val_pred_best.npy") for s in SEEDS], 0)
    po, ps = ens("oct_train"), ens("slo_train"); lo, ls = lg(po), lg(ps)
    fuse = {"probability": lambda w, i: (1 - w) * po[i] + w * ps[i], "logit": lambda w, i: (1 - w) * lo[i] + w * ls[i]}
    rng = np.random.default_rng(20261028)  # same seed and fold construction as e05c_crossfit_weight.py
    res = {(k, w): [] for k in fuse for w in (0.0, 0.15, 0.5)}
    for rep in range(20):
        fold = np.zeros(len(y), int)
        for lab in (0, 1):
            idx = rng.permutation(np.where(y == lab)[0]); fold[idx[: len(idx) // 2]] = 1
        for f in (0, 1):
            te = fold == f; base = roc_auc_score(y[te], po[te])
            for (k, w) in res: res[(k, w)].append(roc_auc_score(y[te], fuse[k](w, te)) - base)
    print("=== TABLE S6: held-out-half AUROC difference vs OCT-only (20 repeats x 2 halves = 40 halves) ===")
    print(f"{'scale':12s} {'w':>5s} {'mean':>8s} {'min':>8s} {'max':>8s} {'halves>0':>9s}")
    for (k, w), d in res.items():
        d = np.array(d); print(f"{k:12s} {w:5.2f} {d.mean():+8.4f} {d.min():+8.4f} {d.max():+8.4f} {(d > 0).sum():6d}/40")
    print("full validation split (no cross-fitting):")
    full = np.ones(len(y), bool); base = roc_auc_score(y, po)
    for (k, w) in res: print(f"  {k:12s} w={w:.2f} delta={roc_auc_score(y, fuse[k](w, full)) - base:+.4f}")


def gamma_oof():
    man = ROOT / "results/e01_dedup/manifests"
    ref = pd.read_csv(man / "fold1_seed42.csv", dtype={"patient_id": str}); ref.patient_id = ref.patient_id.str.zfill(4)
    assert len(ref) == 99 and ref.patient_id.is_unique
    lab = ref.set_index("patient_id").label.astype(int); ids = sorted(lab.index)
    yb = (lab.loc[ids].to_numpy() > 0).astype(int)
    print("manifest columns:", list(ref.columns), "-> check here which GAMMA partition the 99 cases came from")
    print(f"=== GAMMA OUT-OF-FOLD, COLLAPSED BINARY SCORE (n=99, glaucoma={yb.sum()}) ===")
    dirs = {"octlast_concat": "results/e03_c/runs/fold{f}_seed{s}_octlast_concat",
            "octlast_cross": "results/e03_c/runs/fold{f}_seed{s}_octlast_cross",
            "oct_attn_frozen": "results/e03_a/runs/fold{f}_seed{s}_n8_oct_attn"}
    for conf, pat in dirs.items():
        per_seed = []
        for s in SEEDS:
            fr = []
            for f in range(1, 6):
                d = pd.read_csv(ROOT / pat.format(f=f, s=s) / "predictions.csv", dtype={"case_id": str}).rename(columns={"case_id": "patient_id"})
                d.patient_id = d.patient_id.str.zfill(4)
                if "split" in d.columns: d = d[d.split == "test"]
                fr.append(d[["patient_id", "p0", "p1", "p2"]])
            o = pd.concat(fr).set_index("patient_id"); assert o.index.is_unique and len(o) == 99, (conf, s, len(o))
            per_seed.append((o.loc[ids, "p1"] + o.loc[ids, "p2"]).to_numpy())
        aus = [roc_auc_score(yb, p) for p in per_seed]; pm = np.mean(per_seed, 0)
        sens = ((pm >= 0.5) & (yb == 1)).sum() / yb.sum(); spec = ((pm < 0.5) & (yb == 0)).sum() / (yb == 0).sum()
        print(f"{conf:16s} OOF AUROC per seed {np.round(aus, 3).tolist()} mean {np.mean(aus):.3f} | "
              f"seed-mean score: AUROC {roc_auc_score(yb, pm):.3f} sens@0.5 {sens:.3f} spec@0.5 {spec:.3f} mean_p {pm.mean():.3f}")
    print("note: n = 99, so these AUROCs are imprecise; report them with that caveat.")


def slo_ablation(batch=32):
    import torch
    import e05b_baselines as B, e05c_fusion as C
    X = np.load(B.CACHE / "validation_x.npy", mmap_mode="r")
    y = pd.read_csv(B.CACHE / "validation_meta.csv").label.to_numpy(int); assert X.shape[0] == len(y) == 1000
    dev = torch.device("cuda"); assert torch.cuda.is_available()
    mt = torch.tensor(B.MEAN, device=dev).view(1, 1, 3, 1, 1); st = torch.tensor(B.STD, device=dev).view(1, 1, 3, 1, 1)
    perm = np.random.default_rng(20261118).permutation(len(y))  # SLO shuffled across patients, OCT kept
    out = {}
    for kind in ("concat", "cross"):
        P = {m: [] for m in ("intact", "zeros", "shuffled")}
        for s in SEEDS:
            name = f"fusion_{kind}_imagenet_train_seed{s}"
            model = C.FusionNet(kind)
            model.load_state_dict(torch.load(B.RUNS / name / "best.pt", map_location="cpu", weights_only=True), strict=True)
            model = model.to(dev).eval()
            for mode in P:
                ps = []
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for i in range(0, len(y), batch):
                        xb = np.array(X[i:i + batch])
                        if mode == "zeros": xb[:, 0] = 0  # black SLO image before normalization
                        if mode == "shuffled": xb[:, 0] = np.stack([np.array(X[j, 0]) for j in perm[i:i + batch]])
                        ps.append(torch.softmax(model(*C.make_pair(xb, False, None, dev, mt, st)).float(), 1)[:, 1].cpu())
                P[mode].append(torch.cat(ps).numpy())
            del model; torch.cuda.empty_cache()
            ref = np.load(B.RUNS / name / "val_pred_best.npy")
            print(f"{name}: max |intact - saved val_pred_best| = {np.abs(P['intact'][-1] - ref).max():.2e} (should be ~0)")
        out[kind] = {m: np.mean(v, 0) for m, v in P.items()}
    po = np.mean([np.load(B.RUNS / f"oct_train_seed{s}" / "val_pred_best.npy") for s in SEEDS], 0)
    rng = np.random.default_rng(20261118); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    idx = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(2000)]
    def d_ci(a, b):
        d = [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in idx]
        return roc_auc_score(y, a) - roc_auc_score(y, b), np.percentile(d, 2.5), np.percentile(d, 97.5)
    print("=== TABLE S7: SLO ABLATION (validation, 3-seed ensembles) ===")
    print(f"OCT-only AUROC = {roc_auc_score(y, po):.4f}")
    for kind, P in out.items():
        print(f"{kind}: intact AUROC = {roc_auc_score(y, P['intact']):.4f}")
        for m in ("zeros", "shuffled"):
            d, lo, hi = d_ci(P[m], P["intact"]); d2, lo2, hi2 = d_ci(P[m], po)
            print(f"  SLO {m:8s} AUROC = {roc_auc_score(y, P[m]):.4f} | vs intact {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | vs OCT-only {d2:+.4f} [{lo2:+.4f}, {hi2:+.4f}]")
    print("interpretation: a small drop when SLO is removed = the fusion model relied little on SLO.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["dates", "md", "s6", "gamma_oof", "slo_ablation"])
    a = ap.parse_args()
    {"dates": dates, "md": md, "s6": s6, "gamma_oof": gamma_oof, "slo_ablation": slo_ablation}[a.stage]()
