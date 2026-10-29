from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score
)

root = Path("results/e01_dedup")
out = Path("results/e02_analysis")
out.mkdir(parents=True, exist_ok=True)

seeds = [42, 43, 44]
modes = ["fundus", "oct", "fusion"]
cols = ["p0", "p1", "p2"]

ref = pd.read_csv(
    root / "manifests/fold1_seed42.csv",
    dtype={"patient_id": str}
)
ref["patient_id"] = ref["patient_id"].str.zfill(4)
labels = ref.set_index("patient_id")["label"].astype(int)

assert len(labels) == 99
assert labels.index.is_unique
assert "0077" not in labels.index

ids = sorted(labels.index)
y = labels.loc[ids].to_numpy()

all_predictions = []
probabilities = {}
results = []

def auc(y, p):
    return roc_auc_score(
        y, p, multi_class="ovr",
        average="macro", labels=[0, 1, 2]
    )

for seed in seeds:
    for mode in modes:
        parts = []

        for fold in range(1, 6):
            name = f"fold{fold}_seed{seed}"

            manifest = pd.read_csv(
                root / "manifests" / f"{name}.csv",
                dtype={"patient_id": str}
            )
            manifest["patient_id"] = (
                manifest["patient_id"].str.zfill(4)
            )

            test_ids = set(
                manifest.loc[
                    manifest["split"] == "test",
                    "patient_id"
                ]
            )

            path = (
                root / "runs" /
                f"{name}_{mode}" / "predictions.csv"
            )

            df = pd.read_csv(
                path, dtype={"case_id": str}
            )
            df["case_id"] = df["case_id"].str.zfill(4)

            assert df["case_id"].is_unique
            assert set(df["case_id"]) == test_ids
            assert np.isfinite(df[cols].to_numpy()).all()
            assert np.allclose(
                df[cols].sum(axis=1), 1, atol=1e-4
            )

            for r in df.itertuples():
                assert int(r.y_true) == labels.loc[r.case_id]

            df["seed"] = seed
            df["mode"] = mode
            df["fold"] = fold
            parts.append(df)

        pooled = pd.concat(parts, ignore_index=True)

        assert len(pooled) == 99
        assert pooled["case_id"].is_unique
        assert set(pooled["case_id"]) == set(ids)

        pooled = pooled.set_index("case_id").loc[ids]
        p = pooled[cols].to_numpy()
        pred = p.argmax(axis=1)

        probabilities[(seed, mode)] = p

        results.append({
            "seed": seed,
            "mode": mode,
            "n": 99,
            "macro_auc": auc(y, p),
            "macro_f1": f1_score(
                y, pred, average="macro",
                zero_division=0
            ),
            "balanced_accuracy":
                balanced_accuracy_score(y, pred)
        })

        all_predictions.append(pooled.reset_index())

metrics = pd.DataFrame(results)
metrics.to_csv(
    out / "dedup_seed_oof_metrics.csv",
    index=False
)

pd.concat(
    all_predictions, ignore_index=True
).to_csv(out / "dedup_all_oof.csv", index=False)

print("=== SEED OOF ===", flush=True)
print(metrics.round(4).to_string(index=False), flush=True)

print("\n=== MODEL SUMMARY ===", flush=True)
print(
    metrics.groupby("mode")[
        ["macro_auc", "macro_f1", "balanced_accuracy"]
    ].agg(["mean", "std"]).round(4),
    flush=True
)

def average_auc(mode, ix):
    return np.mean([
        auc(y[ix], probabilities[(seed, mode)][ix])
        for seed in seeds
    ])

full = np.arange(99)
observed = {
    mode: average_auc(mode, full)
    for mode in modes
}

rng = np.random.default_rng(20261026)
strata = [
    np.flatnonzero(y == c)
    for c in range(3)
]

draws = {
    "fundus": [],
    "oct": []
}

print("\nRunning 2000 paired bootstrap samples...",
      flush=True)

for b in range(2000):
    ix = np.concatenate([
        rng.choice(g, size=len(g), replace=True)
        for g in strata
    ])

    fusion_auc = average_auc("fusion", ix)

    for baseline in ["fundus", "oct"]:
        draws[baseline].append(
            fusion_auc - average_auc(baseline, ix)
        )

summary = []

print("\n=== PAIRED BOOTSTRAP 95% CI ===", flush=True)

for baseline in ["fundus", "oct"]:
    d = np.array(draws[baseline])
    lo, hi = np.percentile(d, [2.5, 97.5])

    difference = (
        observed["fusion"] - observed[baseline]
    )

    print(
        f"Fusion - {baseline}: "
        f"Delta={difference:+.4f}, "
        f"95% CI=[{lo:+.4f}, {hi:+.4f}]",
        flush=True
    )

    np.save(
        out / f"dedup_bootstrap_fusion_vs_{baseline}.npy",
        d
    )

    summary.append({
        "comparison": f"fusion_vs_{baseline}",
        "auc_difference": difference,
        "ci_low": lo,
        "ci_high": hi,
        "bootstrap_replicates": 2000
    })

pd.DataFrame(summary).to_csv(
    out / "dedup_bootstrap_summary.csv",
    index=False
)

print("\nE02 DEDUP ANALYSIS COMPLETE", flush=True)
print("OOF coverage: 99 cases per seed/model")
print("Saved:", out.resolve())
