import time
import numpy as np
from PIL import Image
import e03b1_slice_policy as b

C32 = np.asarray([
    32*k + j for k in range(8)
    for j in (4, 12, 20, 28)
])

def select_fast(paths, candidates):
    n = len(candidates)
    thumbs = np.empty((n, 64, 64), dtype=np.float32)

    for j, index in enumerate(candidates):
        with Image.open(paths[int(index)]) as im:
            im.draft("L", (64, 64))
            thumbs[j] = np.asarray(
                im.convert("L").resize(
                    (64, 64), Image.Resampling.BILINEAR
                ), dtype=np.float32
            ) / 255.0

    sharp = np.abs(np.diff(thumbs, axis=1)).mean((1,2))
    sharp += np.abs(np.diff(thumbs, axis=2)).mean((1,2))

    flat = thumbs.reshape(n, -1)
    contrast = (
        np.percentile(flat, 95, axis=1)
        - np.percentile(flat, 5, axis=1)
    )
    coverage = (flat > 0.2).mean(axis=1)

    q = (
        0.55 * b.rank01(sharp)
        + 0.30 * b.rank01(contrast)
        + 0.15 * b.rank01(coverage)
    )

    desc = thumbs[:, ::8, ::8].reshape(n, -1)
    desc -= desc.mean(axis=1, keepdims=True)
    desc /= np.maximum(
        np.linalg.norm(desc, axis=1, keepdims=True),
        1e-6
    )

    quality, diverse = [], []

    for k in range(8):
        group = np.flatnonzero(candidates // 32 == k)

        quality.append(
            int(group[np.argmax(q[group])])
        )

        qual = b.rank01(q[group])

        if not diverse:
            best = int(group[np.argmax(qual)])
        else:
            similarity = np.max(
                np.abs(desc[group] @ desc[diverse].T),
                axis=1
            )
            merit = (
                0.7 * qual
                + 0.3 * b.rank01(1 - similarity)
            )
            best = int(group[np.argmax(merit)])

        diverse.append(best)

    return (
        [int(candidates[k]) for k in quality],
        [int(candidates[k]) for k in diverse]
    )

reference = b.read_selection()
rows = b.get_rows().sort_values("patient_id").head(12)

for name, candidates in [
    ("DRAFT256", np.arange(256)),
    ("DRAFT32", C32)
]:
    times = []
    overlaps = []

    for r in rows.itertuples(index=False):
        paths = b.slice_paths(r)

        start = time.perf_counter()
        quality, diverse = select_fast(paths, candidates)
        times.append(
            (time.perf_counter() - start) * 1000
        )

        old = reference[r.patient_id]
        overlaps.append([
            len(set(quality) & set(old["quality8"])),
            len(set(diverse) & set(old["diverse8"]))
        ])

    times = np.asarray(times)
    overlaps = np.asarray(overlaps)

    print(
        name,
        "median_ms", round(np.median(times), 1),
        "p90_ms", round(np.percentile(times, 90), 1),
        "quality_overlap", round(overlaps[:,0].mean(), 2),
        "diverse_overlap", round(overlaps[:,1].mean(), 2),
        flush=True
    )

print("FAST SELECTOR PILOT COMPLETE")
