from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

ROOT = Path("results/e03_c")
RUNS = ROOT / "runs"
MANIFESTS = Path("results/e01_dedup/manifests")
OUT = ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 43, 44)
CONFIGS = (
    "frozen_concat",
    "octlast_concat",
    "bothlast_concat"
)
PAIRS = (
    ("octlast_concat", "frozen_concat"),
    ("bothlast_concat", "octlast_concat"),
    ("bothlast_concat", "frozen_concat")
)
PROB = ["p0", "p1", "p2"]

ref = pd.read_csv(
    MANIFESTS / "fold1_seed42.csv",
    dtype={"patient_id": str}
)
ref.patient_id = ref.patient_id.str.zfill(4)

assert len(ref) == 99 and ref.patient_id.is_unique
labels = ref.set_index("patient_id").label.astype(int)
ids = sorted(labels.index)
y = labels.loc[ids].to_numpy()

P = {}
records = []
fold_records = []

for seed in SEEDS:
    for config in CONFIGS:
        frames = []

        for fold in range(1, 6):
            name = f"fold{fold}_seed{seed}"

            manifest = pd.read_csv(
                MANIFESTS / f"{name}.csv",
                dtype={"patient_id": str}
            )
            manifest.patient_id = (
                manifest.patient_id.str.zfill(4)
            )

            expected = set(
                manifest.loc[
                    manifest.split == "test",
                    "patient_id"
                ]
            )

            folder = RUNS / f"{name}_{config}"
            assert (folder / "complete.json").exists(), folder

            m = json.loads(
                (folder / "metrics.json").read_text()
            )

            fold_records.append({
                "fold": fold,
                "seed": seed,
                "config": config,
                "auc": m["macro_auc_ovr"],
                "macro_f1": m["macro_f1"],
                "balanced_accuracy": m["balanced_accuracy"],
                "best_epoch": m["best_epoch"]
            })

            df = pd.read_csv(
                folder / "predictions.csv",
                dtype={"case_id": str}
            )
            df.case_id = df.case_id.str.zfill(4)

            assert df.case_id.is_unique
            assert set(df.case_id) == expected
            assert np.isfinite(df[PROB].to_numpy()).all()
            assert np.allclose(
                df[PROB].sum(axis=1), 1, atol=1e-4
            )
            assert all(
                int(r.y_true) == int(labels.loc[r.case_id])
                for r in df.itertuples()
            )

            frames.append(df)

        pooled = pd.concat(frames, ignore_index=True)

        assert len(pooled) == 99
        assert pooled.case_id.is_unique
        assert set(pooled.case_id) == set(ids)

        p = pooled.set_index("case_id").loc[
            ids, PROB
        ].to_numpy(float)

        P[config, seed] = p
        pred = p.argmax(axis=1)

        records.append({
            "config": config,
            "seed": seed,
            "macro_auc": roc_auc_score(
                y, p, multi_class="ovr",
                average="macro", labels=[0,1,2]
            ),
            "macro_f1": f1_score(
                y, pred, labels=[0,1,2],
                average="macro", zero_division=0
            ),
            "balanced_accuracy":
                balanced_accuracy_score(y, pred)
        })

fold_df = pd.DataFrame(fold_records)
fold_df.to_csv(
    OUT / "fold_metrics.csv", index=False
)

wide = fold_df.pivot(
    index=["fold","seed"],
    columns="config",
    values="auc"
)

print("=== FOLD-LEVEL PAIRED DIFFERENCES ===")

for a, b in PAIRS:
    d = wide[a] - wide[b]
    print(
        f"{a} - {b}: "
        f"delta={d.mean():+.4f}, "
        f"positive={(d > 0).sum()}/15"
    )

metrics = pd.DataFrame(records)
metrics.to_csv(
    OUT / "seed_oof_metrics.csv", index=False
)

print("\n=== POOLED OOF SUMMARY ===")
print(
    metrics.groupby("config")[
        ["macro_auc","macro_f1","balanced_accuracy"]
    ].agg(["mean","std"]).round(4)
)

def auc(yy, pp):
    scores = []
    for c in (0,1,2):
        pos = yy == c
        n1 = int(pos.sum())
        n0 = len(yy) - n1
        ranks = rankdata(pp[:,c], method="average")
        scores.append(
            (ranks[pos].sum() - n1*(n1+1)/2) / (n1*n0)
        )
    return float(np.mean(scores))

def mean_auc(config, ix):
    return np.mean([
        auc(y[ix], P[config,seed][ix])
        for seed in SEEDS
    ])

rng = np.random.default_rng(20261027)
strata = [
    np.flatnonzero(y == c)
    for c in (0,1,2)
]

B = 2000
boot = {pair: [] for pair in PAIRS}
full = np.arange(len(y))

print("\nBOOTSTRAP 2000...", flush=True)

for _ in range(B):
    ix = np.concatenate([
        rng.choice(s, len(s), replace=True)
        for s in strata
    ])

    values = {
        c: mean_auc(c, ix)
        for c in CONFIGS
    }

    for a, b in PAIRS:
        boot[a,b].append(values[a] - values[b])

summary = []

print("\n=== PAIRED BOOTSTRAP 95% CI ===")

for a, b in PAIRS:
    delta = mean_auc(a, full) - mean_auc(b, full)
    lo, hi = np.percentile(
        boot[a,b], [2.5,97.5]
    )

    print(
        f"{a} - {b}: "
        f"delta={delta:+.4f}, "
        f"CI=[{lo:+.4f},{hi:+.4f}]"
    )

    summary.append({
        "comparison": f"{a}_minus_{b}",
        "delta": delta,
        "ci_low": lo,
        "ci_high": hi,
        "bootstrap_samples": B
    })

pd.DataFrame(summary).to_csv(
    OUT / "paired_bootstrap.csv", index=False
)

print("\nE03-C OOF AUDIT COMPLETE")
print("Saved:", OUT)
