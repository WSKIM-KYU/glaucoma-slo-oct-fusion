import argparse, glob, io, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data/external/fairvision/Glaucoma/Dataset/dataset.zip"
PAIRS = ROOT / "results/e04/fairvision_pairs_audit.csv"
OUT = ROOT / "results/e05/audit/input_stats"; OUT.mkdir(parents=True, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--oct-axis", type=int, default=0)
ap.add_argument("--slices", type=int, default=8)
ap.add_argument("--gamma-fundus", default=None)
ap.add_argument("--gamma-oct", default=None)
args = ap.parse_args()

def to_u8(a):
    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float64)
    return ((a - a.min()) / (a.max() - a.min() + 1e-9) * 255).round().astype(np.uint8)

def model_input(a):
    pic = Image.fromarray(to_u8(a)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    return (np.asarray(pic, dtype=np.float64) / 255.0 - MEAN) / STD

class Acc:
    def __init__(s): s.raw, s.m, s.sd, s.dt = [], [], [], set()
    def add(s, a):
        a = np.asarray(a); s.dt.add(str(a.dtype)); f = a.astype(np.float64)
        s.raw.append([f.mean(), f.std(), np.percentile(f, 1), np.percentile(f, 99)])
        x = model_input(a).reshape(-1, 3); s.m.append(x.mean(0)); s.sd.append(x.std(0))
    def out(s):
        r = np.array(s.raw)
        return {"n": len(r), "dtype": ",".join(sorted(s.dt)),
                "raw_mean": round(float(r[:, 0].mean()), 2), "raw_std": round(float(r[:, 1].mean()), 2),
                "raw_p1": round(float(r[:, 2].mean()), 2), "raw_p99": round(float(r[:, 3].mean()), 2),
                "norm_mean_rgb": np.round(np.mean(s.m, 0), 3).tolist(),
                "norm_std_rgb": np.round(np.mean(s.sd, 0), 3).tolist()}

pairs = pd.read_csv(PAIRS, dtype={"case_id": str})
split = pairs["split"].astype(str).str.strip().str.lower()
val = pairs[split == "validation"].sort_values("case_id").reset_index(drop=True)
assert len(val) == 1000, len(val)
picks = val.iloc[np.linspace(0, 999, min(args.n, 1000)).round().astype(int)]

slo_acc, oct_acc = Acc(), Acc()
with zipfile.ZipFile(ZIP) as z:
    for _, r in picks.iterrows():
        with np.load(io.BytesIO(z.read(r["npz_path"])), allow_pickle=False) as d:
            slo_acc.add(d["slo_fundus"]); vol = np.asarray(d["oct_bscans"])
        n = vol.shape[args.oct_axis]
        for i in np.linspace(0, n - 1, args.slices).round().astype(int):
            oct_acc.add(np.take(vol, i, axis=args.oct_axis))

res = {"fairvision_val_slo": slo_acc.out(), "fairvision_val_oct": oct_acc.out()}
for name, pat in (("gamma_fundus", args.gamma_fundus), ("gamma_oct", args.gamma_oct)):
    if pat:
        files = sorted(glob.glob(pat, recursive=True)); assert files, f"no files: {pat}"
        acc = Acc()
        for f in files[: args.n * (args.slices if name == "gamma_oct" else 1)]:
            acc.add(np.asarray(Image.open(f)))
        res[name] = acc.out()

(OUT / "input_stats.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("=== E05 STEP 2a: INPUT DISTRIBUTION AUDIT ===")
print(f"OCT axis = {args.oct_axis}, slices = {args.slices}")
for k, v in res.items():
    print(f"{k:20s} {v}")
print("E05 STEP 2a OUTPUT SAVED:", OUT)
