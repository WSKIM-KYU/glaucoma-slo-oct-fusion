"""Post hoc supplementary statistics for the E05 analysis-locked test evaluation.

Re-creates the locked bootstrap resamples (same seed and order), checks that the 95% CIs match
results.json, then reports Bonferroni/Holm adjustment, partial AUROC (FPR <= 0.2) and sensitivity at a
test-matched specificity of 0.85. results.json and the lock are not modified. Run from repository root.
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "e05")
import e05_formal_test as T

proto = json.loads(T.PROTOCOL.read_text()); res = json.loads(T.RESULT.read_text()); cal = proto["calibration"]
y = pd.read_csv(T.CACHE / "test_meta.csv").label.to_numpy(int)
Z = T.cand_logits(T.ens(lambda n: np.load(T.PRED / f"{n}.npy")))
P = {c: T.sig(cal[c]["platt_a"] * Z[c] + cal[c]["platt_b"]) for c in T.CANDIDATES}
C = T.CANDIDATES
rng = np.random.default_rng(T.BOOT_SEED)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
auc, pauc, sens = ({c: [] for c in C} for _ in range(3))

def sens_at_spec(yy, p, spec=0.85):
    thr = np.quantile(p[yy == 0], spec)
    return np.mean(p[yy == 1] > thr)

for k in range(T.N_BOOT):
    i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    for c in C:
        auc[c].append(roc_auc_score(y[i], P[c][i]))
        pauc[c].append(roc_auc_score(y[i], P[c][i], max_fpr=0.2))
        sens[c].append(sens_at_spec(y[i], P[c][i]))
    if (k + 1) % 1000 == 0: print(f"BOOTSTRAP {k+1}/{T.N_BOOT}", flush=True)
auc, pauc, sens = ({c: np.array(v) for c, v in d.items()} for d in (auc, pauc, sens))

print("\n=== [A] reproduction check vs results.json (95% CI) ===")
for c in ("cross", "concat", "late_w015"):
    ci = np.percentile(auc[c] - auc["oct"], [2.5, 97.5]); ref = res["deltas"][f"{c}_minus_oct"]["ci"]
    ok = np.allclose(ci, ref, atol=1e-12)
    print(f"  {c:10s} reproduced={ci.round(6).tolist()} locked={np.round(ref, 6).tolist()} {'OK' if ok else 'MISMATCH'}")
    assert ok, "bootstrap not reproduced; stop"

def pval(d): return min(1.0, 2 * max(min(np.mean(d <= 0), np.mean(d >= 0)), 1 / (len(d) + 1)))
def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, j in enumerate(order):
        run = max(run, (len(ps) - r) * ps[j]); adj[j] = min(1.0, run)
    return adj

names = ["cross", "concat", "late_w015"]
D = {c: auc[c] - auc["oct"] for c in names}
p = np.array([pval(D[c]) for c in names])
h_all = holm(p); h_sec = np.r_[np.nan, holm(p[1:])]
print("\n=== [B] multiplicity (AUROC difference vs OCT-only) ===")
out = {}
for j, c in enumerate(names):
    d0 = res["models"][c]["auroc"] - res["models"]["oct"]["auroc"]
    cis = {q: np.percentile(D[c], [(100 - q) / 2, 100 - (100 - q) / 2]) for q in (95, 97.5, 98.333)}
    print(f"  {c:10s} delta={d0:+.4f} 95%={cis[95].round(4).tolist()} 97.5%={cis[97.5].round(4).tolist()} "
          f"98.33%={cis[98.333].round(4).tolist()} p={p[j]:.4f} holm(2)={h_sec[j]:.4f} holm(3)={h_all[j]:.4f}")
    out[c] = {"delta": d0, "ci95": cis[95].tolist(), "ci97_5": cis[97.5].tolist(), "ci98_33": cis[98.333].tolist(),
              "p_boot": p[j], "holm_secondary2": None if np.isnan(h_sec[j]) else h_sec[j], "holm_all3": h_all[j]}

print("\n=== [C] partial AUROC (FPR<=0.2, McClish-standardized) ===")
pa = {}
for c in C:
    base = roc_auc_score(y, P[c], max_fpr=0.2); pa[c] = base
    line = f"  {c:10s} pAUC={base:.4f} [{np.percentile(pauc[c], 2.5):.4f},{np.percentile(pauc[c], 97.5):.4f}]"
    if c != "oct":
        dd = pauc[c] - pauc["oct"]
        line += f" delta={base - roc_auc_score(y, P['oct'], max_fpr=0.2):+.4f} [{np.percentile(dd, 2.5):+.4f},{np.percentile(dd, 97.5):+.4f}]"
    print(line)

print("\n=== [D] sensitivity at test-matched specificity 0.85 (descriptive) ===")
sm = {}
for c in C:
    base = sens_at_spec(y, P[c]); sm[c] = base
    line = f"  {c:10s} sens={base:.4f} [{np.percentile(sens[c], 2.5):.4f},{np.percentile(sens[c], 97.5):.4f}]"
    if c != "oct":
        dd = sens[c] - sens["oct"]
        line += f" delta={base - sens_at_spec(y, P['oct']):+.4f} [{np.percentile(dd, 2.5):+.4f},{np.percentile(dd, 97.5):+.4f}]"
    print(line)

(T.OUT / "supplementary_stats.json").write_text(json.dumps({
    "note": "post hoc supplementary statistics; same bootstrap seed and resamples as the locked evaluation; results.json unchanged",
    "multiplicity": out, "pauc_fpr0_2": pa, "sens_at_test_spec085": sm}, indent=2, default=float), encoding="utf-8")
print("\nSUPPLEMENTARY STATS DONE")
