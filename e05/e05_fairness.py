#!/usr/bin/env python
"""E05 subgroup fairness analysis (descriptive, post-hoc). Uses locked Test predictions and calibration; no refitting."""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05_formal_test as T
import e05b_baselines as B

OUT = T.OUT
N_BOOT, SEED, MIN_N, MIN_CLASS = 2000, 20261030, 30, 10
MODELS = T.CANDIDATES

assert T.PROTOCOL.exists() and T.RESULT.exists(), "run formal evaluate first"
proto = json.loads(T.PROTOCOL.read_text()); res = json.loads(T.RESULT.read_text())
meta = pd.read_csv(T.CACHE / "test_meta.csv", dtype={"case_id": str})
y = meta.label.to_numpy(int)
Z = T.cand_logits(T.ens(lambda n: np.load(T.PRED / f"{n}.npy")))
cal = proto["calibration"]
P = {c: T.sig(cal[c]["platt_a"] * Z[c] + cal[c]["platt_b"]) for c in MODELS}
for c in MODELS:
    assert abs(roc_auc_score(y, P[c]) - res["models"][c]["auroc"]) < 1e-9, f"mismatch with results.json: {c}"
print("=== E05 SUBGROUP FAIRNESS (descriptive, FairVision Test 3000) ===")
print("consistency with results.json: OK")

sm = pd.read_csv(T.SUMMARY_CSV, dtype={"filename": str})
sm["case_id"] = sm.filename.str.extract(r"(\d{5})\.npz$")[0]
print("summary CSV columns:", list(sm.columns))
sm = sm.set_index("case_id").loc[meta.case_id]
attrs = {}
for col in ("race", "gender", "ethnicity", "language", "maritalstatus"):
    if col in sm.columns:
        attrs[col] = sm[col].astype(str).str.strip().str.lower().to_numpy()
if "age" in sm.columns:
    age = pd.to_numeric(sm["age"], errors="coerce")
    attrs["age_group"] = pd.cut(age, [-np.inf, 50, 60, 70, np.inf], right=False,
                                labels=["<50", "50-59", "60-69", "70+"]).astype(str).to_numpy()
if not {"race", "gender", "ethnicity"} <= set(attrs):
    print("reading missing attributes from npz (race, male, hispanic) ...", flush=True)
    pairs = pd.read_csv(T.ROOT / "results/e04/fairvision_pairs_audit.csv", dtype={"case_id": str}).set_index("case_id")
    codes = {"race": [], "male": [], "hispanic": []}
    with zipfile.ZipFile(B.ZIP) as z:
        for i, cid in enumerate(meta.case_id):
            with np.load(io.BytesIO(z.read(pairs.loc[cid, "npz_path"])), allow_pickle=False) as d:
                for k in codes: codes[k].append(int(d[k]))
            if (i + 1) % 1000 == 0: print(f"  npz attributes {i+1}/3000", flush=True)
    attrs.setdefault("race", np.array([f"race_code{v}" for v in codes["race"]]))
    attrs.setdefault("gender", np.array(["male" if v == 1 else "female" for v in codes["male"]]))
    attrs.setdefault("ethnicity", np.array(["hispanic" if v == 1 else "non-hispanic" for v in codes["hispanic"]]))
    print("NOTE: npz codes assumed male=1, hispanic=1; race codes unmapped (compare counts with dataset paper)")

rng = np.random.default_rng(SEED)
def rates(p, yy, thr):
    yh = p >= thr
    tpr = float((yh & (yy == 1)).sum() / max((yy == 1).sum(), 1))
    fpr = float((yh & (yy == 0)).sum() / max((yy == 0).sum(), 1))
    return tpr, 1 - fpr, fpr, float(yh.mean())

rows, summary = [], {}
for attr, g in attrs.items():
    print(f"\n--- {attr} ---")
    groups = [v for v in pd.Series(g).value_counts().index]
    valid = {}
    for v in groups:
        m = g == v; yy = y[m]
        if m.sum() < MIN_N or (yy == 1).sum() < MIN_CLASS or (yy == 0).sum() < MIN_CLASS:
            print(f"  {v:16s} n={m.sum():5d}  too small, excluded from AUROC")
            continue
        valid[v] = np.where(m)[0]
    if len(valid) < 2:
        print("  fewer than 2 analysable groups"); continue
    print(f"  {'group':16s} {'n':>5s} {'prev':>5s} | " + " | ".join(f"{c:>21s}" for c in MODELS) + " | late-oct dAUROC [95%CI]")
    per = {c: {} for c in MODELS}
    for v, idx in valid.items():
        yy = y[idx]; pos, neg = idx[yy == 1], idx[yy == 0]
        bs = {c: [] for c in MODELS}
        for _ in range(N_BOOT):
            i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            for c in MODELS: bs[c].append(roc_auc_score(y[i], P[c][i]))
        bs = {c: np.array(b) for c, b in bs.items()}
        cells = []
        for c in MODELS:
            auc = roc_auc_score(yy, P[c][idx]); lo, hi = np.percentile(bs[c], [2.5, 97.5])
            s5, sp5, f5, pr5 = rates(P[c][idx], yy, 0.5)
            s85, sp85, f85, pr85 = rates(P[c][idx], yy, cal[c]["thr_spec85_val"])
            per[c][v] = {"auc": auc, "tpr5": s5, "fpr5": f5, "pos5": pr5, "tpr85": s85, "fpr85": f85}
            cells.append(f"{auc:.3f} [{lo:.3f},{hi:.3f}]")
            rows.append({"attribute": attr, "group": v, "model": c, "n": len(idx), "prevalence": float(yy.mean()),
                         "auroc": auc, "auroc_lo": lo, "auroc_hi": hi, "mean_p": float(P[c][idx].mean()),
                         "sens_0p5": s5, "spec_0p5": sp5, "sens_thr85": s85, "spec_thr85": sp85})
        d = bs["late_w015"] - bs["oct"]
        dd = roc_auc_score(yy, P["late_w015"][idx]) - roc_auc_score(yy, P["oct"][idx])
        dl, dh = np.percentile(d, [2.5, 97.5])
        print(f"  {v:16s} {len(idx):5d} {yy.mean():5.3f} | " + " | ".join(cells) + f" | {dd:+.4f} [{dl:+.4f},{dh:+.4f}]")
    print(f"\n  {'model':10s} {'worstAUC':>8s} {'maxGap':>7s} {'ES-AUC':>7s} {'DPD@.5':>7s} {'DEOdds@.5':>9s} {'DEOdds@thr85':>12s}")
    summary[attr] = {}
    for c in MODELS:
        aucs = np.array([per[c][v]["auc"] for v in valid])
        allidx = np.concatenate(list(valid.values()))
        auc_all = roc_auc_score(y[allidx], P[c][allidx])
        es = auc_all / (1 + np.abs(auc_all - aucs).sum())
        pos5 = [per[c][v]["pos5"] for v in valid]
        deo5 = max(np.ptp([per[c][v]["tpr5"] for v in valid]), np.ptp([per[c][v]["fpr5"] for v in valid]))
        deo85 = max(np.ptp([per[c][v]["tpr85"] for v in valid]), np.ptp([per[c][v]["fpr85"] for v in valid]))
        summary[attr][c] = {"worst_group_auroc": float(aucs.min()), "max_auroc_gap": float(np.ptp(aucs)),
                            "es_auc": float(es), "dpd_0p5": float(np.ptp(pos5)),
                            "deodds_0p5": float(deo5), "deodds_thr85": float(deo85), "groups": list(valid)}
        print(f"  {c:10s} {aucs.min():8.4f} {np.ptp(aucs):7.4f} {es:7.4f} {np.ptp(pos5):7.4f} {deo5:9.4f} {deo85:12.4f}")

pd.DataFrame(rows).to_csv(OUT / "fairness_subgroups.csv", index=False)
(OUT / "fairness_summary.json").write_text(json.dumps(
    {"note": "descriptive post-hoc analysis; locked calibration and thresholds; no model or threshold changes",
     "bootstrap": {"replicates": N_BOOT, "seed": SEED}, "min_group": {"n": MIN_N, "per_class": MIN_CLASS},
     "summary": summary}, indent=2), encoding="utf-8")
print(f"\nSAVED {OUT / 'fairness_subgroups.csv'} and fairness_summary.json")
print("E05 FAIRNESS ANALYSIS DONE")
