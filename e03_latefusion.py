from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score
)

src = Path("results/e02_analysis/dedup_all_oof.csv")
out = Path("results/e03_latefusion")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src, dtype={"case_id": str})
df["case_id"] = df["case_id"].str.zfill(4)

seeds = (42, 43, 44)
modes = ("fundus", "oct", "fusion")
cols = ["p0", "p1", "p2"]

assert len(df) == 891
assert not df.duplicated(["seed", "mode", "case_id"]).any()

P = {}
Y = None
ids = None
folds = None

for seed in seeds:
    for mode in modes:
        x = df[
            (df.seed == seed) & (df["mode"] == mode)
        ].set_index("case_id").sort_index()

        assert len(x) == 99
        assert "0077" not in x.index

        y = x.y_true.to_numpy(dtype=int)
        f = x.fold.to_numpy(dtype=int)
        p = x[cols].to_numpy(dtype=float)

        assert np.isfinite(p).all()
        assert (p >= 0).all()
        assert np.allclose(p.sum(1), 1, atol=1e-4)

        if Y is None:
            Y = y
            ids = x.index.to_numpy()
            folds = f
            assert set(Y) == {0, 1, 2}

        assert np.array_equal(ids, x.index)
        assert np.array_equal(Y, y)
        assert np.array_equal(folds, f)

        P[seed, mode] = p

    P[seed, "late50"] = (
        P[seed, "fundus"] + P[seed, "oct"]
    ) / 2

def auc(y, p):
    values = []
    for k in range(3):
        pos = y == k
        n1 = int(pos.sum())
        n0 = len(y) - n1
        r = rankdata(p[:, k], method="average")
        values.append(
            (r[pos].sum() - n1*(n1+1)/2) / (n1*n0)
        )
    return float(np.mean(values))

assert abs(
    auc(Y, P[42, "fusion"]) -
    roc_auc_score(
        Y, P[42, "fusion"],
        multi_class="ovr", average="macro"
    )
) < 1e-12

rows = []
preds = []

for seed in seeds:
    for mode in (*modes, "late50"):
        p = P[seed, mode]
        pred = p.argmax(1)

        rows.append({
            "seed": seed,
            "mode": mode,
            "n": len(Y),
            "macro_auc": auc(Y, p),
            "macro_f1": f1_score(
                Y, pred, average="macro",
                zero_division=0
            ),
            "balanced_accuracy":
                balanced_accuracy_score(Y, pred)
        })

    p = P[seed, "late50"]

    for i, cid in enumerate(ids):
        preds.append({
            "seed": seed,
            "case_id": cid,
            "fold": int(folds[i]),
            "y_true": int(Y[i]),
            "y_pred": int(p[i].argmax()),
            **dict(zip(cols, p[i]))
        })

res = pd.DataFrame(rows)
res.to_csv(out / "seed_metrics.csv", index=False)
pd.DataFrame(preds).to_csv(
    out / "late50_oof.csv", index=False
)

print("=== E03 SEED RESULTS ===")
print(res.round(4).to_string(index=False))

print("\n=== MODEL SUMMARY ===")
print(
    res.groupby("mode")[
        ["macro_auc", "macro_f1",
         "balanced_accuracy"]
    ].agg(["mean", "std"]).round(4)
)

def mean_auc(mode, ix):
    return np.mean([
        auc(Y[ix], P[seed, mode][ix])
        for seed in seeds
    ])

rng = np.random.default_rng(20261026)
strata = [
    np.flatnonzero(Y == k)
    for k in range(3)
]

draws = {"oct": [], "fusion": []}

print("\nRunning 2000 bootstrap samples...")

for _ in range(2000):
    ix = np.concatenate([
        rng.choice(z, size=len(z), replace=True)
        for z in strata
    ])

    late = mean_auc("late50", ix)

    for base in draws:
        draws[base].append(
            late - mean_auc(base, ix)
        )

summary = []
full = np.arange(len(Y))

print("\n=== BOOTSTRAP 95% CI ===")

for base in draws:
    delta = (
        mean_auc("late50", full) -
        mean_auc(base, full)
    )

    lo, hi = np.percentile(
        draws[base], [2.5, 97.5]
    )

    print(
        f"Late50 - {base}: "
        f"delta={delta:+.4f}, "
        f"95% CI=[{lo:+.4f}, {hi:+.4f}]"
    )

    summary.append({
        "comparison": f"late50_vs_{base}",
        "auc_difference": delta,
        "ci_low": lo,
        "ci_high": hi
    })

pd.DataFrame(summary).to_csv(
    out / "bootstrap_summary.csv",
    index=False
)

print("\nE03-1 COMPLETE")
