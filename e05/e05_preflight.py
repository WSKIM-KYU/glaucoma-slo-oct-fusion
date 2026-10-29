from pathlib import Path
import json
import pandas as pd

BASE = Path("data/external/fairvision")
CSV = BASE / "Glaucoma/ReadMe/data_summary_glaucoma.csv"
PAIRS = Path("results/e04/fairvision_pairs_audit.csv")
OUT = Path("results/e05/audit")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV, dtype={"filename": str})
pairs = pd.read_csv(PAIRS, dtype={"case_id": str})

df["case_id"] = df["filename"].str.extract(r"(\d{5})\.npz$")[0]
df["split"] = df["use"].str.strip().str.lower()
df["label"] = df["glaucoma"].str.strip().str.lower().map({
    "no": 0,
    "yes": 1
})

assert len(df) == 10000
assert df["case_id"].notna().all()
assert df["case_id"].is_unique
assert df["label"].notna().all()

assert df["split"].value_counts().to_dict() == {
    "training": 6000,
    "validation": 1000,
    "test": 3000
}

assert len(pairs) == 10000
assert not pairs.duplicated(["split", "case_id"]).any()

merged = df.merge(
    pairs,
    on=["case_id", "split"],
    how="left",
    validate="one_to_one"
)

assert merged["npz_path"].notna().all()
assert merged["jpg_path"].notna().all()

train = merged[merged["split"] == "training"].copy()
val = merged[merged["split"] == "validation"].copy()
test = merged[merged["split"] == "test"].copy()

assert set(train.case_id).isdisjoint(val.case_id)
assert set(train.case_id).isdisjoint(test.case_id)
assert set(val.case_id).isdisjoint(test.case_id)

columns = ["case_id", "split", "label", "npz_path", "jpg_path"]

train[columns].to_csv(
    OUT / "e05_train_manifest.csv", index=False
)
val[columns].to_csv(
    OUT / "e05_val_manifest.csv", index=False
)

# No Test labels exported to E05 training manifests.

summary = {
    "training_cases": len(train),
    "validation_cases": len(val),
    "test_cases_excluded": len(test),
    "training_labels": {
        str(k): int(v)
        for k, v in train.label.value_counts().sort_index().items()
    },
    "validation_labels": {
        str(k): int(v)
        for k, v in val.label.value_counts().sort_index().items()
    },
    "test_used_for_training": False,
    "test_used_for_model_selection": False
}

(OUT / "e05_preflight.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("=== E05 DATA PREFLIGHT ===")
print(json.dumps(summary, indent=2))
print("E05 PREFLIGHT PASS")
