#!/usr/bin/env python3
"""E03-B1 unsupervised OCT slice-policy comparison on frozen ResNet18 features.

Requires e03a_volume_mil.py and its feature cache alongside this script.
This is exploratory development, not a patient-level independent validation.
"""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

import e03a_volume_mil as old

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "e03_b1"
SELECT_FILE = OUTPUT / "selections.json"
MODES = ("uniform8", "random8", "quality8", "diverse8")
SEEDS = (42, 43, 44)
FOLDS = tuple(range(1, 6))
BINS = tuple(np.array_split(np.arange(256), 8))
QUALITY_WEIGHTS = (0.55, 0.30, 0.15)


def get_rows():
    df = old.manifest(1, 42).copy()
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(4)
    assert len(df) == 99 and df.patient_id.is_unique
    assert "0077" not in set(df.patient_id)
    return df


def slice_paths(r):
    oct_dir = Path(r.oct_path).parent
    paths = sorted(oct_dir.glob("*_image.jpg"), key=lambda p: int(p.stem.split("_")[0]))
    assert len(paths) == 256, oct_dir
    assert len({p.name for p in paths}) == 256
    return paths


def thumb_matrix(paths):
    arr = np.empty((256, 64, 64), dtype=np.float32)
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            arr[i] = np.asarray(
                im.convert("L").resize((64, 64), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
    return arr


def rank01(x):
    x = np.asarray(x, dtype=np.float64)
    return (rankdata(x, method="average") - 1.0) / (len(x) - 1)


def quality_scores(thumbs):
    # Cheap, label-free image proxies, not validated clinical quality scores.
    sharp = np.abs(np.diff(thumbs, axis=1)).mean((1, 2)) + np.abs(
        np.diff(thumbs, axis=2)
    ).mean((1, 2))
    flat = thumbs.reshape(256, -1)
    contrast = np.percentile(flat, 95, axis=1) - np.percentile(flat, 5, axis=1)
    coverage = (flat > 0.2).mean(axis=1)
    return (QUALITY_WEIGHTS[0] * rank01(sharp)
            + QUALITY_WEIGHTS[1] * rank01(contrast)
            + QUALITY_WEIGHTS[2] * rank01(coverage))


def select_positions(case_id, paths, modes=MODES):
    indices = {}
    if "uniform8" in modes:
        indices["uniform8"] = old.get_positions(8).tolist()
    if "random8" in modes:
        seed = int.from_bytes(hashlib.sha256(("e03b1:" + case_id).encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        indices["random8"] = [int(rng.choice(b)) for b in BINS]
    if "quality8" in modes or "diverse8" in modes:
        thumbs = thumb_matrix(paths)
        q = quality_scores(thumbs)
        if "quality8" in modes:
            indices["quality8"] = [int(b[np.argmax(q[b])]) for b in BINS]
        if "diverse8" in modes:
            # Predefined low-cost quality + sequential appearance diversity rule.
            desc = thumbs[:, ::8, ::8].reshape(256, -1)
            desc = desc - desc.mean(axis=1, keepdims=True)
            norm = np.linalg.norm(desc, axis=1, keepdims=True).clip(min=1e-6)
            desc = desc / norm
            selected = []
            for b in BINS:
                qual = rank01(q[b])
                if not selected:
                    best = int(b[np.argmax(qual)])
                else:
                    sim = np.max(np.abs(desc[b] @ desc[selected].T), axis=1)
                    novelty = 1.0 - sim
                    merit = 0.7 * qual + 0.3 * rank01(novelty)
                    best = int(b[np.argmax(merit)])
                selected.append(best)
            indices["diverse8"] = selected
    for k, pos in indices.items():
        assert len(pos) == 8 and len(set(pos)) == 8 and pos == sorted(pos), (case_id, k, pos)
    return indices


def select_stage():
    if SELECT_FILE.exists():
        print("EXISTS (not overwritten):", SELECT_FILE)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = {}
    rows = get_rows()
    for i, r in enumerate(rows.itertuples(index=False), 1):
        selection[r.patient_id] = select_positions(r.patient_id, slice_paths(r))
        if i % 10 == 0 or i == len(rows):
            print("SELECTED", i, "/", len(rows), flush=True)
    payload = {
        "protocol": "E03-B1-v1; no labels used by selection; 8 spatial bins",
        "methods": list(MODES), "n_cases": len(selection),
        "quality_weights": list(QUALITY_WEIGHTS), "cases": selection,
    }
    SELECT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SELECTION PASS: 99 cases x 4 methods x 8 slices")
    print("SAVED:", SELECT_FILE)


def read_selection():
    if not SELECT_FILE.is_file():
        raise FileNotFoundError("Run --stage select first: " + str(SELECT_FILE))
    data = json.loads(SELECT_FILE.read_text(encoding="utf-8"))
    assert set(data["cases"]) == set(get_rows().patient_id)
    assert set(data["methods"]) == set(MODES)
    for cid, by_method in data["cases"].items():
        for mode in MODES:
            idx = by_method[mode]
            assert len(idx) == 8 and len(set(idx)) == 8 and idx == sorted(idx), (cid, mode)
            assert min(idx) >= 0 and max(idx) < 256
    return data["cases"]


class PhysicalPositionHead(old.Head):
    """Same layers as E03-A Head; use actual normalized B-scan indices."""
    def __init__(self):
        super().__init__("gated")

    def forward(self, fundus, octs, slice_indices):
        pos = (slice_indices.to(octs.dtype) * (2.0 / 255.0) - 1.0).unsqueeze(-1)
        weight = torch.softmax(
            self.att(torch.cat([octs, pos], dim=-1)).squeeze(-1), dim=1
        )
        ov = (weight.unsqueeze(-1) * octs).sum(dim=1)
        o = self.oct_proj(ov)
        f = self.fundus_proj(fundus)
        gate = torch.sigmoid(self.gate(torch.cat([o, f], dim=-1)))
        return self.classifier(gate * f + (1 - gate) * o)


def experiment_one(features, selection, fold, seed, method, epochs, batch):
    exp = f"fold{fold}_seed{seed}_{method}_gated"
    dest = OUTPUT / "runs" / exp
    complete = dest / "complete.json"
    if complete.exists():
        for name in ("best.pt", "predictions.csv", "metrics.json"):
            if not (dest / name).is_file():
                raise RuntimeError("Incomplete completed experiment: " + str(dest))
        print("SKIP", exp, flush=True)
        return

    part = old.manifest(fold, seed).copy()
    part["patient_id"] = part["patient_id"].astype(str).str.zfill(4)
    assert len(part) == 99 and part.patient_id.is_unique

    def collect(split):
        subset = part[part["split"] == split]
        cases = subset.patient_id.tolist()
        f = torch.stack([features[c]["fundus"] for c in cases]).float()
        ind = torch.tensor([selection[c][method] for c in cases], dtype=torch.long)
        octs = torch.stack([features[c]["oct"][ind[i]] for i, c in enumerate(cases)]).float()
        labels = torch.tensor(subset.label.to_numpy(dtype=int), dtype=torch.long)
        return cases, f, octs, ind, labels

    tr, va, te = [collect(s) for s in ("train", "val", "test")]
    assert not (set(tr[0]) & set(va[0]) or set(tr[0]) & set(te[0]) or set(va[0]) & set(te[0]))
    old.set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicalPositionHead().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    counts = np.bincount(tr[4].numpy(), minlength=3)
    weights = torch.tensor(len(tr[4]) / (3 * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(TensorDataset(tr[1], tr[2], tr[3], tr[4]), batch_size=batch, shuffle=True)
    vf, vo, vi, vy = [x.to(dev) for x in va[1:]]
    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for f, o, ind, yy in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(f.to(dev), o.to(dev), ind.to(dev))
            loss = criterion(pred, yy.to(dev))
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            vl = float(criterion(model(vf, vo, vi), vy).item())
        if vl < best_loss - 1e-5:
            best_loss = vl
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, wait = epoch, 0
        else:
            wait += 1
        if wait >= 12:
            break
    model.load_state_dict(best_state)
    model.eval()
    tf, to, ti, ty = [x.to(dev) for x in te[1:]]
    with torch.inference_mode():
        prob = torch.softmax(model(tf, to, ti), dim=1).cpu().numpy()
    y = te[4].numpy()
    yh = prob.argmax(axis=1)
    try:
        auroc = float(roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except ValueError:
        auroc = None
    metrics = {
        "experiment": exp, "fold": fold, "seed": seed, "method": method,
        "n_slices": 8, "n": len(y), "n_train": len(tr[0]), "n_val": len(va[0]),
        "best_epoch": best_epoch, "val_loss": best_loss, "macro_auc_ovr": auroc,
        "macro_f1": float(f1_score(y, yh, labels=[0, 1, 2], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yh)),
        "accuracy": float(accuracy_score(y, yh)),
        "confusion_matrix": confusion_matrix(y, yh, labels=[0, 1, 2]).tolist(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"case_id": cid, "y_true": int(y[i]), "y_pred": int(yh[i]),
         "p0": float(prob[i, 0]), "p1": float(prob[i, 1]), "p2": float(prob[i, 2])}
        for i, cid in enumerate(te[0])
    ]).to_csv(dest / "predictions.csv", index=False)
    torch.save(best_state, dest / "best.pt")
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    complete.write_text(json.dumps({"complete": True, "experiment": exp}), encoding="utf-8")
    print("COMPLETE", exp, "AUROC", auroc, "best_epoch", best_epoch, flush=True)


def train_stage(methods, epochs, batch, limit_runs):
    select = read_selection()
    features = old.load_features()
    count = 0
    for fold in FOLDS:
        for seed in SEEDS:
            for method in methods:
                if limit_runs and count >= limit_runs:
                    print("STOP --limit-runs", limit_runs)
                    return
                experiment_one(features, select, fold, seed, method, epochs, batch)
                count += 1
    print("E03-B1 TRAIN COMPLETE")


def summary_stage():
    records = []
    for path in (OUTPUT / "runs").glob("*/metrics.json"):
        if not (path.parent / "complete.json").is_file():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        records.append(rec)
    if not records:
        raise RuntimeError("No complete runs")
    df = pd.DataFrame(records)
    print("RUNS", len(df), "/60")
    print("\n=== FOLD x SEED SUMMARY ===")
    print(df.groupby("method")[["macro_auc_ovr", "macro_f1", "balanced_accuracy"]].agg(["count", "mean", "std"]).round(4))
    wide = df.pivot_table(index=["fold", "seed"], columns="method", values="macro_auc_ovr")
    print("\n=== METHOD - UNIFORM8 (MATCHED PAIRS) ===")
    for method in MODES[1:]:
        if method not in wide.columns or "uniform8" not in wide.columns:
            continue
        d = (wide[method] - wide["uniform8"]).dropna()
        if len(d):
            print(method, "mean", round(d.mean(), 4), "positive", int((d > 0).sum()), "/", len(d))
    df.to_csv(OUTPUT / "run_metrics.csv", index=False)


def latency_stage():
    # Selector overhead only. Include this in later end-to-end latency comparison.
    rows = get_rows().sort_values("patient_id").head(12)
    measurements = []
    for r in rows.itertuples(index=False):
        paths = slice_paths(r)
        for _ in range(2):
            t0 = time.perf_counter()
            select_positions(r.patient_id, paths, modes=("quality8", "diverse8"))
            measurements.append(1000 * (time.perf_counter() - t0))
    a = np.array(measurements)
    print("THUMBNAIL SCORING + SELECTOR", "n", len(a), "median_ms", round(float(np.median(a)), 1), "p90_ms", round(float(np.percentile(a, 90)), 1))
    print("NOTE: add this overhead to raw JPEG -> ResNet18 -> Fusion inference; not a full latency benchmark")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("select", "train", "summary", "selector-latency"), required=True)
    ap.add_argument("--methods", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit-runs", type=int, default=0)
    args = ap.parse_args()
    if args.stage == "select":
        select_stage()
    elif args.stage == "train":
        train_stage(args.methods, args.epochs, args.batch, args.limit_runs)
    elif args.stage == "summary":
        summary_stage()
    else:
        latency_stage()


if __name__ == "__main__":
    main()
