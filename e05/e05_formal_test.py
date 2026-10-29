#!/usr/bin/env python
"""E05 formal Test evaluation: fixed candidates, locked protocol, single evaluation on FairVision Test 3000.
FairVision Test was used once in E04 for GAMMA-trained models only. E05 models never used Test."""
import argparse, hashlib, io, json, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05b_baselines as B
import e05c_fusion as C

ROOT, RUNS, CACHE, log = B.ROOT, B.RUNS, B.CACHE, B.log
OUT = ROOT / "results/e05/formal_test"; PRED = OUT / "test_preds"
PROTOCOL, RESULT = OUT / "protocol.json", OUT / "results.json"
SUMMARY_CSV = ROOT / "data/external/fairvision/Glaucoma/ReadMe/data_summary_glaucoma.csv"
SEEDS = (42, 43, 44)
RUNSETS = {"oct": "oct_train_seed{}", "slo": "slo_train_seed{}",
           "cross": "fusion_cross_imagenet_train_seed{}", "concat": "fusion_concat_imagenet_train_seed{}"}
CANDIDATES = ("oct", "cross", "concat", "late_w015")
W_SLO, N_BOOT, BOOT_SEED, TARGET_SPEC = 0.15, 5000, 20261029, 0.85

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
sig = lambda z: 1 / (1 + np.exp(-z))

def ens(src):
    return {k: np.mean([src(v.format(s)) for s in SEEDS], 0) for k, v in RUNSETS.items()}

def cand_logits(P):
    z = {k: lg(P[k]) for k in ("oct", "cross", "concat")}
    z["late_w015"] = (1 - W_SLO) * lg(P["oct"]) + W_SLO * lg(P["slo"])
    return z

def ece(y, p, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def lock():
    if PROTOCOL.exists(): raise SystemExit("protocol already locked: " + str(PROTOCOL))
    OUT.mkdir(parents=True, exist_ok=True)
    yv = pd.read_csv(CACHE / "validation_meta.csv").label.to_numpy(int)
    hashes = {}
    for v in RUNSETS.values():
        for s in SEEDS:
            d = RUNS / v.format(s)
            assert (d / "best.pt").is_file() and (d / "val_pred_best.npy").is_file(), d
            hashes[d.name] = sha(d / "best.pt")
    Z = cand_logits(ens(lambda n: np.load(RUNS / n / "val_pred_best.npy")))
    cal = {}
    for c in CANDIDATES:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(Z[c].reshape(-1, 1), yv)
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        pv = sig(a * Z[c] + b)
        thr = float(np.quantile(pv[yv == 0], TARGET_SPEC))
        cal[c] = {"platt_a": a, "platt_b": b, "thr_spec85_val": thr, "val_auroc": float(roc_auc_score(yv, pv))}
    proto = {
        "version": "e05_formal_test_v1", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": "Harvard-FairVision Glaucoma official Test 3000 (reused: evaluated once in E04 for GAMMA-trained models; E05 models never used Test)",
        "reference": "oct (OCT-only, 3-seed probability-mean ensemble)",
        "primary_endpoint": "AUROC(cross) - AUROC(oct), paired stratified bootstrap 95% CI",
        "secondary": ["AUROC(concat) - AUROC(oct)", "AUROC(late_w015) - AUROC(oct)",
                      "AUPRC, Brier, ECE(10 bins), sens/spec at 0.5 and at val spec85 threshold"],
        "candidates": {"oct": RUNSETS["oct"], "cross": RUNSETS["cross"], "concat": RUNSETS["concat"],
                       "late_w015": f"(1-{W_SLO})*logit(oct) + {W_SLO}*logit(slo), weight fixed from validation cross-fit"},
        "ensemble": "mean of seed probabilities 42,43,44, then logit",
        "calibration": cal, "bootstrap": {"replicates": N_BOOT, "seed": BOOT_SEED, "scheme": "stratified by label, paired"},
        "input": "SLO + 8 B-scans (axis0, uniform of 200), 224 bilinear, ImageNet normalization",
        "checkpoint_sha256": hashes, "script_sha256": sha(Path(__file__)),
        "rule": "single evaluation; no model, weight, calibration or threshold change after viewing Test"}
    PROTOCOL.write_text(json.dumps(proto, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"LOCKED {PROTOCOL}")
    for c in CANDIDATES: log(f"  {c:10s} platt a={cal[c]['platt_a']:.3f} b={cal[c]['platt_b']:+.3f} thr85={cal[c]['thr_spec85_val']:.3f}")

def check_lock():
    if not PROTOCOL.exists(): raise SystemExit("run --stage lock first")
    p = json.loads(PROTOCOL.read_text())
    if p["script_sha256"] != sha(Path(__file__)): raise SystemExit("script changed after lock")
    for name, h in p["checkpoint_sha256"].items():
        if sha(RUNS / name / "best.pt") != h: raise SystemExit("checkpoint changed after lock: " + name)
    return p

def prepare():
    check_lock()
    out, meta = CACHE / "test_x.npy", CACHE / "test_meta.csv"
    if out.exists() and meta.exists(): log("SKIP test cache exists"); return
    pairs = pd.read_csv(ROOT / "results/e04/fairvision_pairs_audit.csv", dtype={"case_id": str})
    t = pairs[pairs.split.str.lower() == "test"].sort_values("case_id").reset_index(drop=True)
    assert len(t) == 3000 and t.case_id.is_unique
    sm = pd.read_csv(SUMMARY_CSV, dtype={"filename": str})
    sm["case_id"] = sm.filename.str.extract(r"(\d{5})\.npz$")[0]
    ref = dict(zip(sm.case_id, sm.glaucoma.str.strip().str.lower().map({"no": 0, "yes": 1})))
    tmp = CACHE / "test_x.partial.npy"
    x = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(3000, 1 + B.N_SLICES, 224, 224))
    labels = np.empty(3000, int)
    with zipfile.ZipFile(B.ZIP) as z:
        for i, r in enumerate(t.itertuples(index=False)):
            with np.load(io.BytesIO(z.read(r.npz_path)), allow_pickle=False) as d:
                slo, vol, lab = np.asarray(d["slo_fundus"]), np.asarray(d["oct_bscans"]), int(d["glaucoma"])
            if lab != ref[r.case_id]: raise RuntimeError("label mismatch " + r.case_id)
            labels[i] = lab
            x[i, 0] = B.to224(slo)
            for j, k in enumerate(B.IDX): x[i, 1 + j] = B.to224(np.take(vol, k, axis=B.BSCAN_AXIS))
            if (i + 1) % 500 == 0: log(f"prepare test {i+1}/3000")
    x.flush(); del x
    import os; os.replace(tmp, out)
    pd.DataFrame({"case_id": t.case_id, "label": labels}).to_csv(meta, index=False)
    log("DONE test cache")

def infer(batch=32):
    check_lock()
    X = np.load(CACHE / "test_x.npy", mmap_mode="r"); assert X.shape == (3000, 9, 224, 224)
    PRED.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda"); assert torch.cuda.is_available()
    mt = torch.tensor(B.MEAN, device=dev).view(1, 1, 3, 1, 1); st = torch.tensor(B.STD, device=dev).view(1, 1, 3, 1, 1)
    for kind, v in RUNSETS.items():
        for s in SEEDS:
            name = v.format(s); out = PRED / f"{name}.npy"
            if out.exists(): log("SKIP " + name); continue
            model = B.OctNet() if kind == "oct" else B.SloNet() if kind == "slo" else C.FusionNet(kind)
            model.load_state_dict(torch.load(RUNS / name / "best.pt", map_location="cpu", weights_only=True), strict=True)
            model = model.to(dev).eval(); ps = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, 3000, batch):
                    xb = np.array(X[i:i+batch])
                    if kind in ("oct", "slo"):
                        logits = model(B.make_input(xb, kind, False, None, dev, mt, st))
                    else:
                        logits = model(*C.make_pair(xb, False, None, dev, mt, st))
                    ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu())
            p = torch.cat(ps).numpy(); assert p.shape == (3000,) and np.isfinite(p).all()
            np.save(out, p); log(f"INFER DONE {name}")
            del model; torch.cuda.empty_cache()

def evaluate():
    proto = check_lock()
    if RESULT.exists(): raise SystemExit("already evaluated: " + str(RESULT))
    y = pd.read_csv(CACHE / "test_meta.csv").label.to_numpy(int); assert len(y) == 3000
    Z = cand_logits(ens(lambda n: np.load(PRED / f"{n}.npy")))
    cal = proto["calibration"]
    P = {c: sig(cal[c]["platt_a"] * Z[c] + cal[c]["platt_b"]) for c in CANDIDATES}
    def ss(p, t):
        yh = (p >= t).astype(int)
        return float(((yh == 1) & (y == 1)).sum() / (y == 1).sum()), float(((yh == 0) & (y == 0)).sum() / (y == 0).sum())
    res = {"protocol_version": proto["version"], "n": 3000, "prevalence": float(y.mean()), "models": {}, "deltas": {}}
    rng = np.random.default_rng(BOOT_SEED); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = {c: [] for c in CANDIDATES}
    for k in range(N_BOOT):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for c in CANDIDATES: boots[c].append(roc_auc_score(y[i], P[c][i]))
        if (k + 1) % 1000 == 0: log(f"BOOTSTRAP {k+1}/{N_BOOT}")
    boots = {c: np.array(v) for c, v in boots.items()}
    for c in CANDIDATES:
        s05, p05 = ss(P[c], 0.5); s85, p85 = ss(P[c], cal[c]["thr_spec85_val"])
        res["models"][c] = {"auroc": float(roc_auc_score(y, P[c])), "auroc_ci": np.percentile(boots[c], [2.5, 97.5]).tolist(),
                            "auprc": float(average_precision_score(y, P[c])), "brier": float(np.mean((P[c] - y) ** 2)),
                            "ece10": ece(y, P[c]), "sens_0p5": s05, "spec_0p5": p05, "sens_thr85": s85, "spec_thr85": p85,
                            "mean_p": float(P[c].mean())}
    for c in ("cross", "concat", "late_w015"):
        d = boots[c] - boots["oct"]
        res["deltas"][f"{c}_minus_oct"] = {"delta": res["models"][c]["auroc"] - res["models"]["oct"]["auroc"],
                                           "ci": np.percentile(d, [2.5, 97.5]).tolist(), "role": "PRIMARY" if c == "cross" else "secondary"}
    RESULT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n=== E05 FORMAL TEST RESULTS (FairVision Test 3000, reused) ===")
    print(f"prevalence = {res['prevalence']:.4f}")
    print(f"{'model':10s} {'AUROC':>7s} {'95% CI':>17s} {'AUPRC':>6s} {'Brier':>6s} {'ECE':>6s} {'sens.5':>6s} {'spec.5':>6s} {'sens85':>6s} {'spec85':>6s}")
    for c, m in res["models"].items():
        print(f"{c:10s} {m['auroc']:7.4f} [{m['auroc_ci'][0]:.4f},{m['auroc_ci'][1]:.4f}] {m['auprc']:6.4f} {m['brier']:6.4f} "
              f"{m['ece10']:6.4f} {m['sens_0p5']:6.3f} {m['spec_0p5']:6.3f} {m['sens_thr85']:6.3f} {m['spec_thr85']:6.3f}")
    print("\nPaired AUROC differences vs OCT-only:")
    for k, d in res["deltas"].items():
        print(f"  {k:20s} {d['delta']:+.4f}  95%CI [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]  {d['role']}")
    print("E05 FORMAL TEST EVALUATION COMPLETE")

def status():
    print("LOCK", PROTOCOL.exists(), "| TEST CACHE", (CACHE / "test_x.npy").exists(),
          "| PREDICTIONS", len(list(PRED.glob("*.npy"))) if PRED.exists() else 0, "/ 12",
          "| EVALUATED", RESULT.exists())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["lock", "prepare", "infer", "evaluate", "status"])
    a = ap.parse_args()
    {"lock": lock, "prepare": prepare, "infer": infer, "evaluate": evaluate, "status": status}[a.stage]()
