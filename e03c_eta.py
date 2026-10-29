from pathlib import Path
from datetime import datetime, timedelta
from statistics import median
import time

files = list(Path("results/e03_c/runs").glob("*/complete.json"))
times = sorted(p.stat().st_mtime for p in files)

done = len(files)
total = 45
remain = max(0, total - done)

print("E03-C FINE-TUNING MONITOR")
print("=" * 45)
print(f"Completed : {done}/{total}")
print(f"Progress  : {100 * done / total:.1f}%")

if done == total:
    print("STATUS    : COMPLETE")
elif done >= 3:
    recent = times[-7:]
    gaps = [
        recent[i] - recent[i-1]
        for i in range(1, len(recent))
    ]
    run_time = median(gaps)
    elapsed = time.time() - times[-1]
    eta = max(0, remain * run_time - min(elapsed, run_time))
    finish = datetime.now() + timedelta(seconds=eta)

    print(f"Run time  : {run_time:.1f} sec")
    print(f"Remaining : {str(timedelta(seconds=int(eta)))}")
    print(f"Finish ETA: {finish:%Y-%m-%d %H:%M:%S}")
else:
    print("ETA available after 3 completed runs")

print("=" * 45)
print("Updated:", datetime.now().strftime("%H:%M:%S"))
