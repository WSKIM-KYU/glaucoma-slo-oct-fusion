#!/usr/bin/env python
"""E05 race-gap decomposition (descriptive, post-hoc): age-standardized AUROC, MD-severity sensitivity, normal-eye FPR.
Locked predictions/calibration/thresholds; no refitting."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e05_formal_test as T

MODELS = ("oct", "late_w015", "concat")
proto = json.loads(T.PROTOCOL.read_text()); cal = proto["calibration"]
meta = pd.read_csv(T.CACHE / "test_meta.csv", dtype={"case_id": str}); y = meta.label.to_numpy(int)
Z = T.cand_logits(T.ens(lambda n: np.load(T.PRED / f"{n}.npy")))
P = {c: T.sig(cal[c]["platt_a"] * Z[c] + cal[c]["platt_b"]) for c in MODELS}
sm = pd.read_csv(T.SUMMARY_CSV, dtype={"filename": str})
sm["case_id"] = sm.filename.str.extract(r"(\d{5})\.npz$")[0]
sm = sm.set_index("case_id").loc[meta.case_id]
race = sm.race.astype(str).str.strip().str.lower().to_numpy()
age = pd.to_numeric(sm.age, errors="coerce").to_numpy()
md = pd.to_numeric(sm.md, errors="coerce").to_numpy()
agrp = pd.cut(age, [-np.inf, 50, 60, 70, np.inf], right=False, labels=["<50", "50-59", "60-69", "70+"]).astype(str)
agrp = np.asarray(agrp)
RACES = [r for r in ("white", "black", "asian") if (race == r).any()]
AGES = ["<50", "50-59", "60-69", "70+"]
print("=== E05 RACE GAP DECOMPOSITION (descriptive) ===")

print("\n[A] age distribution by race (row %), mean age, prevalence")
for r in RACES:
    m = race == r
    dist = [100 * np.mean(agrp[m] == a) for a in AGES]
    print(f"  {r:6s} n={m.sum():5d} " + " ".join(f"{a}:{d:4.1f}%" for a, d in zip(AGES, dist))
          + f" | mean_age={np.nanmean(age[m]):5.1f} prev={y[m].mean():.3f}")

print("\n[B] AUROC within age strata, and age-standardized AUROC (weights = overall age distribution)")
w_all = {a: np.mean(agrp == a) for a in AGES}
for c in MODELS:
    print(f"  model={c}")
    for r in RACES:
        cells, num, den = [], 0.0, 0.0
        for a in AGES:
            m = (race == r) & (agrp == a)
            if m.sum() >= 30 and 10 <= y[m].sum() <= m.sum() - 10:
                au = roc_auc_score(y[m], P[c][m]); cells.append(f"{a}:{au:.3f}(n{m.sum()})")
                num += w_all[a] * au; den += w_all[a]
            else:
                cells.append(f"{a}:  -  (n{m.sum()})")
        crude = roc_auc_score(y[race == r], P[c][race == r])
        std = num / den if den > 0 else float("nan")
        print(f"    {r:6s} crude={crude:.3f} age_std={std:.3f} (strata covered {den:.2f}) | " + "  ".join(cells))

print("\n[C] glaucoma eyes: sensitivity by MD severity (HPA: early >= -6, moderate -12..-6, advanced < -12 dB)")
sev = np.full(len(y), "", dtype=object)
sev[(y == 1) & (md >= -6)] = "early"
sev[(y == 1) & (md < -6) & (md >= -12)] = "moderate"
sev[(y == 1) & (md < -12)] = "advanced"
print(f"  glaucoma eyes with MD: {np.isfinite(md[y == 1]).sum()}/{(y == 1).sum()}")
for r in RACES:
    m = (race == r) & (y == 1)
    mix = " ".join(f"{s}:{100*np.mean(sev[m] == s):4.1f}%" for s in ("early", "moderate", "advanced"))
    print(f"  {r:6s} glaucoma n={m.sum():4d} median MD={np.nanmedian(md[m]):6.2f} | {mix}")
for c in MODELS:
    thr = cal[c]["thr_spec85_val"]
    print(f"  model={c} sensitivity at locked thr85={thr:.3f}")
    for r in RACES:
        cells = []
        for s in ("early", "moderate", "advanced"):
            m = (race == r) & (sev == s)
            cells.append(f"{s}:{np.mean(P[c][m] >= thr):.3f}(n{m.sum()})" if m.sum() >= 15 else f"{s}:  -  (n{m.sum()})")
        print(f"    {r:6s} " + "  ".join(cells))

print("\n[D] normal eyes: false-positive rate and mean predicted probability by race")
for c in MODELS:
    thr = cal[c]["thr_spec85_val"]
    cells = []
    for r in RACES:
        m = (race == r) & (y == 0)
        cells.append(f"{r}: FPR@thr85={np.mean(P[c][m] >= thr):.3f} FPR@.5={np.mean(P[c][m] >= .5):.3f} mean_p={P[c][m].mean():.3f} (n{m.sum()})")
    print(f"  model={c}\n    " + "\n    ".join(cells))
print("\nE05 RACE GAP DECOMPOSITION DONE")
