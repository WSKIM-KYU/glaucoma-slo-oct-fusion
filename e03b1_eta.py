from pathlib import Path
from datetime import datetime, timedelta
from statistics import median
import time

ROOT = Path("results/e03_b1/runs")
TOTAL = 60

files = list(ROOT.glob("*/complete.json"))
times = sorted(p.stat().st_mtime for p in files)
done = len(files)
remain = max(0, TOTAL - done)

now = time.time()
pct = min(done / TOTAL, 1) * 100

print("E03-B1 TRAINING MONITOR")
print("=" * 44)
print(f"Completed       : {done} / {TOTAL}")
print(f"Progress        : {pct:.1f}%")
print(f"Remaining runs  : {remain}")

bar = "#" * int(pct / 5)
bar += "-" * (20 - len(bar))
print(f"[{bar}]")

def fmt(sec):
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

if done >= TOTAL:
    print("\nSTATUS: ALL EXPERIMENTS COMPLETE")

elif done >= 3:
    recent = times[-min(done, 7):]
    gaps = [
        recent[i] - recent[i - 1]
        for i in range(1, len(recent))
    ]

    avg = median(gaps)
    since = now - times[-1]

    if avg <= 0:
        print("\nETA unavailable: invalid timestamps")
    else:
        eta = max(0, remain * avg - min(since, avg))
        finish = datetime.now() + timedelta(seconds=eta)

        print(f"Recent run time : {avg:.1f} sec/run")
        print(f"Remaining ETA   : {fmt(eta)}")
        print(f"Expected finish : {finish:%Y-%m-%d %H:%M:%S}")

        if since > max(180, 3 * avg):
            print("\nWARNING: No recent completion.")
            print("Check train.log and the training process.")

else:
    print("\nCollecting run-time data...")
    print("ETA available after at least 3 completions.")

print("=" * 44)
print("Updated:", datetime.now().strftime("%H:%M:%S"))
