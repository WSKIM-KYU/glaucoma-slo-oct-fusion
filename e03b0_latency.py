import time
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from e03a_volume_mil import (
    BASE, Head, get_positions,
    manifest, weights_and_transforms
)

assert torch.cuda.is_available(), "CUDA GPU required"

project = Path(BASE)
out = project / "results/e03_b0"
out.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(4)
device = torch.device("cuda:0")

encoder, transform = weights_and_transforms()
encoder = encoder.eval().to(device)

rows = list(
    manifest(1, 42)
    .sort_values("patient_id")
    .head(12)
    .itertuples(index=False)
)

records = []

@torch.inference_mode()
def infer(paths, head):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    features = []

    for k in range(0, len(paths), 16):
        images = []
        for path in paths[k:k+16]:
            with Image.open(path) as im:
                images.append(
                    transform(im.convert("RGB"))
                )

        batch = torch.stack(images).to(device)
        features.append(encoder(batch))

    z = torch.cat(features)
    probs = torch.softmax(
        head(z[:1], z[1:].unsqueeze(0)), dim=1
    )

    assert torch.isfinite(probs).all()
    torch.cuda.synchronize()

    ms = 1000 * (time.perf_counter() - start)
    mem = torch.cuda.max_memory_allocated() / 1048576
    return ms, mem

for n in [8, 256]:
    checkpoint = project / (
        f"results/e03_a/runs/"
        f"fold1_seed42_n{n}_gated/best.pt"
    )

    assert checkpoint.is_file(), checkpoint

    head = Head("gated").eval().to(device)
    state = torch.load(
        checkpoint, map_location="cpu",
        weights_only=True
    )
    head.load_state_dict(state)

    positions = get_positions(n)

    for i, r in enumerate(rows):
        slices = sorted(
            Path(r.oct_path).parent.glob("*_image.jpg"),
            key=lambda p: int(p.stem.split("_")[0])
        )

        assert len(slices) == 256

        paths = [Path(r.fundus_path)] + [
            slices[int(j)] for j in positions
        ]

        if i == 0:
            infer(paths, head)

        for rep in range(2):
            ms, mem = infer(paths, head)

            records.append({
                "slices": n,
                "case_id": str(r.patient_id).zfill(4),
                "repeat": rep,
                "latency_ms": ms,
                "peak_gpu_allocated_mib": mem
            })

    part = pd.DataFrame(
        [x for x in records if x["slices"] == n]
    )

    print(
        f"n={n}: "
        f"median={part.latency_ms.median():.1f}ms, "
        f"p90={part.latency_ms.quantile(.9):.1f}ms, "
        f"GPU={part.peak_gpu_allocated_mib.max():.1f}MiB",
        flush=True
    )

df = pd.DataFrame(records)
df.to_csv(out / "latency_raw.csv", index=False)

summary = df.groupby("slices").agg(
    measurements=("latency_ms", "size"),
    median_ms=("latency_ms", "median"),
    p90_ms=(
        "latency_ms",
        lambda x: x.quantile(.9)
    ),
    peak_gpu_allocated_mib=(
        "peak_gpu_allocated_mib", "max"
    )
)

summary.to_csv(out / "latency_summary.csv")

print("\n=== E03-B0 BENCHMARK ===")
print(summary.round(2))
print("E03-B0 COMPLETE")
