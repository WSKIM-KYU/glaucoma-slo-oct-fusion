"""E05-C cross-fitted SLO weight for late fusion (validation, 2-fold x 20 repeats). Run from repository root."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
R = Path("results/e05/runs")
y = pd.read_csv("results/e05/cache/validation_meta.csv").label.to_numpy(int)
ens = lambda n: np.mean([np.load(R / f"{n}_seed{s}" / "val_pred_best.npy") for s in (42, 43, 44)], 0)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
lo, ls = lg(ens("oct_train")), lg(ens("slo_train"))
W = np.round(np.arange(0, 0.51, 0.05), 2)
chosen, deltas = [], []
rng = np.random.default_rng(20261028)
for rep in range(20):
    fold = np.zeros(len(y), int)
    for lab in (0, 1):
        idx = rng.permutation(np.where(y == lab)[0]); fold[idx[: len(idx) // 2]] = 1
    pred = np.zeros(len(y))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        best = max(W, key=lambda w: roc_auc_score(y[tr], (1 - w) * lo[tr] + w * ls[tr]))
        chosen.append(best)
        pred[te] = (1 - best) * lo[te] + best * ls[te]
    deltas.append(roc_auc_score(y, pred) - roc_auc_score(y, lo))
deltas = np.array(deltas)
print("=== E05-C CROSS-FITTED LATE FUSION (Validation, 2-fold x 20 repeats) ===")
print("SLO weight grid:", W.tolist())
print("chosen SLO weights: median", np.median(chosen), "range", min(chosen), "-", max(chosen))
print(f"OCT-only AUROC       = {roc_auc_score(y, lo):.4f}")
print(f"cross-fit delta AUROC: mean {deltas.mean():+.4f}, min {deltas.min():+.4f}, max {deltas.max():+.4f}")
print(f"repeats with delta > 0: {(deltas > 0).sum()}/20")
print("E05-C CROSS-FIT DONE")
