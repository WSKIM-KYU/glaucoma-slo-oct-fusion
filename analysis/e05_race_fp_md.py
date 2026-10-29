"""Normal eyes: false positives vs true negatives by race (OCT-only, locked threshold). Run from repository root."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "e05")
import e05_formal_test as T
proto = json.loads(T.PROTOCOL.read_text()); cal = proto["calibration"]
meta = pd.read_csv(T.CACHE / "test_meta.csv", dtype={"case_id": str}); y = meta.label.to_numpy(int)
Z = T.cand_logits(T.ens(lambda n: np.load(T.PRED / f"{n}.npy")))
p = T.sig(cal["oct"]["platt_a"] * Z["oct"] + cal["oct"]["platt_b"]); thr = cal["oct"]["thr_spec85_val"]
sm = pd.read_csv(T.SUMMARY_CSV, dtype={"filename": str})
sm["case_id"] = sm.filename.str.extract(r"(\d{5})\.npz$")[0]; sm = sm.set_index("case_id").loc[meta.case_id]
race = sm.race.astype(str).str.lower().to_numpy(); md = pd.to_numeric(sm.md, errors="coerce").to_numpy()
age = pd.to_numeric(sm.age, errors="coerce").to_numpy()
print("=== NORMAL EYES: false positives vs true negatives (OCT-only, locked thr85) ===")
for r in ("white", "black", "asian"):
    m = (race == r) & (y == 0); fp = m & (p >= thr); tn = m & (p < thr)
    f = lambda a, k: f"{np.nanmedian(a[k]):6.2f}" if k.sum() else "   -  "
    print(f"{r:6s} FP n={fp.sum():4d} median MD={f(md, fp)} age={f(age, fp)} | "
          f"TN n={tn.sum():4d} median MD={f(md, tn)} age={f(age, tn)} | "
          f"MD<-2dB among FP={np.mean(md[fp] < -2) if fp.sum() else float('nan'):.2f} TN={np.mean(md[tn] < -2):.2f}")
print("DONE")
