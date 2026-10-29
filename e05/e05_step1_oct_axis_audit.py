import io, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data/external/fairvision/Glaucoma/Dataset/dataset.zip"
PAIRS = ROOT / "results/e04/fairvision_pairs_audit.csv"
OUT = ROOT / "results/e05/audit/oct_axis"; OUT.mkdir(parents=True, exist_ok=True)
N = 6

def to_u8(a):
    a = np.asarray(a, dtype=np.float64)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)

def band_ratio(img):
    img = img.astype(np.float64)
    return float(img.mean(1).var() / (img.mean(0).var() + 1e-9))

def best_corr(proj, slo):
    s = slo.astype(np.float64)
    p = np.asarray(Image.fromarray(to_u8(proj)).resize(s.shape[::-1], Image.Resampling.BILINEAR), dtype=np.float64)
    best = (-2.0, "")
    for k in range(4):
        for f in (False, True):
            q = np.rot90(p, k)
            q = np.fliplr(q) if f else q
            if q.shape != s.shape:
                continue
            c = float(np.corrcoef(q.ravel(), s.ravel())[0, 1])
            if c > best[0]:
                best = (c, f"rot{90*k}" + ("+flip" if f else ""))
    return best

pairs = pd.read_csv(PAIRS, dtype={"case_id": str})
split = pairs["split"].astype(str).str.strip().str.lower()
train = pairs[split == "training"].sort_values("case_id").reset_index(drop=True)
assert len(train) == 6000, len(train)
picks = train.iloc[np.linspace(0, len(train) - 1, N).round().astype(int)]

rows = []
with zipfile.ZipFile(ZIP) as z:
    names = set(z.namelist())
    missing = [p for p in picks["npz_path"] if p not in names]
    assert not missing, f"not in zip: {missing[:3]}"
    for _, r in picks.iterrows():
        with np.load(io.BytesIO(z.read(r["npz_path"])), allow_pickle=False) as d:
            keys = {k: (tuple(d[k].shape), str(d[k].dtype)) for k in d.files}
            vol = np.asarray(d["oct_bscans"]).astype(np.float64)
            slo = np.asarray(d["slo_fundus"])
        if slo.ndim == 3:
            slo = slo.mean(-1)
        sd = vol.std() + 1e-9
        tiles = []
        for a in range(3):
            mid = np.take(vol, vol.shape[a] // 2, axis=a)
            proj = vol.mean(axis=a)
            c, tf = best_corr(proj, slo)
            jump = float(np.abs(np.diff(vol, axis=a)).mean() / sd)
            rows.append({"case": r["case_id"], "axis": a, "slice_shape": mid.shape,
                         "band_rows": band_ratio(mid), "band_cols": band_ratio(mid.T),
                         "proj_slo_corr": c, "proj_tf": tf, "adjacent_jump": jump})
            for im in (mid, proj):
                tiles.append(np.asarray(Image.fromarray(to_u8(im)).resize((200, 200))))
        tiles.append(np.asarray(Image.fromarray(to_u8(slo)).resize((200, 200))))
        Image.fromarray(np.hstack(tiles)).save(OUT / f"case_{r['case_id']}.png")

df = pd.DataFrame(rows)
df.to_csv(OUT / "oct_axis_metrics.csv", index=False)
print("=== E05 STEP 1: OCT AXIS AUDIT ===")
print("NPZ keys (last case):")
for k, v in keys.items():
    print(f"  {k}: shape={v[0]} dtype={v[1]}")
print("\nPer-case metrics:")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
m = df.groupby("axis")[["band_rows", "band_cols", "proj_slo_corr", "adjacent_jump"]].mean()
print("\nMean over cases:")
print(m.to_string(float_format=lambda x: f"{x:.3f}"))
depth = int(m["proj_slo_corr"].idxmax())
lateral = [a for a in range(3) if a != depth]
bscan = int(m.loc[lateral, "adjacent_jump"].idxmax())
print(f"\nSUGGESTED depth axis   = {depth}  (projection most similar to SLO)")
print(f"SUGGESTED B-scan index = {bscan}  (larger slow-axis discontinuity)")
print(f"E04 used axis 0 -> {'B-scan (consistent)' if bscan == 0 else 'NOT B-scan index: review needed'}")
print("PNG tiles: [ax0 slice, ax0 proj, ax1 slice, ax1 proj, ax2 slice, ax2 proj, SLO]")
print("E05 STEP 1 OUTPUT SAVED:", OUT)
