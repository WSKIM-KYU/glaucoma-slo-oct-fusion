"""E05-C late-fusion diagnostic on validation (no fitted weights). Run from repository root."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
R = Path("results/e05/runs")
y = pd.read_csv("results/e05/cache/validation_meta.csv").label.to_numpy(int)
def ens(name): return np.mean([np.load(R / f"{name}_seed{s}" / "val_pred_best.npy") for s in (42, 43, 44)], 0)
oct_p, slo_p, cross_p = ens("oct_train"), ens("slo_train"), ens("fusion_cross_imagenet_train")
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
late_prob = (oct_p + slo_p) / 2
late_logit = 1 / (1 + np.exp(-(logit(oct_p) + logit(slo_p)) / 2))
print("=== E05-C LATE FUSION DIAGNOSTIC (Validation, no fitted weights) ===")
print(f"corr(OCT, SLO) pearson   = {np.corrcoef(oct_p, slo_p)[0,1]:.4f}")
print(f"corr(OCT, SLO) rank      = {pd.Series(oct_p).corr(pd.Series(slo_p), method='spearman'):.4f}")
for lab, t in ((0, "normal"), (1, "glaucoma")):
    m = y == lab
    print(f"within-{t:8s} rank corr = {pd.Series(oct_p[m]).corr(pd.Series(slo_p[m]), method='spearman'):.4f}")
rng = np.random.default_rng(20261028); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
def delta(p, ref, B=2000):
    d = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], ref[i]))
    return roc_auc_score(y, p) - roc_auc_score(y, ref), np.percentile(d, 2.5), np.percentile(d, 97.5)
print(f"\n{'model':28s} {'AUROC':>7s} {'delta_vs_OCT':>13s}  95% CI")
print(f"{'OCT-only':28s} {roc_auc_score(y, oct_p):7.4f}")
for name, p in (("late fusion (prob mean)", late_prob), ("late fusion (logit mean)", late_logit), ("joint Cross-Attention", cross_p)):
    d, lo, hi = delta(p, oct_p)
    print(f"{name:28s} {roc_auc_score(y, p):7.4f} {d:+13.4f}  [{lo:+.4f}, {hi:+.4f}]")
print("E05-C LATE FUSION DONE")
