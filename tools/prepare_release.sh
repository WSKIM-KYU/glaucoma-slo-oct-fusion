#!/bin/bash
# Collect the original study scripts and aggregate results into this repository.
# Usage (from the repository root):  bash tools/prepare_release.sh [/path/to/OpticNeuroFusion_Starter]
# Originals are copied byte-for-byte so that the SHA-256 recorded in the locked protocol can be verified.
set -euo pipefail
SRC="${1:-/home/test/OpticNeuroFusion_Starter}"
DST="$(pwd)"
[ -f "$DST/README.md" ] || { echo "Run from the repository root"; exit 1; }
[ -d "$SRC" ] || { echo "Project directory not found: $SRC"; exit 1; }

copy() {  # copy <relative source> <relative destination>
  if [ -f "$SRC/$1" ]; then
    mkdir -p "$DST/$(dirname "$2")"; cp -p "$SRC/$1" "$DST/$2"; echo "  + $2"
  else
    echo "  ! missing $1"
  fi
}

echo "[1/5] Study scripts (byte-identical copies)"
for f in e03a_volume_mil.py e03c_train.py e04_full_external.py; do copy "$f" "$f"; done
for f in "$SRC"/e0[0-3]*.py; do
  b="$(basename "$f")"; case "$b" in e03a_volume_mil.py|e03c_train.py) ;; *) [ -f "$f" ] && copy "$b" "$b" ;; esac
done
for f in "$SRC"/e05/*.py; do copy "e05/$(basename "$f")" "e05/$(basename "$f")"; done

echo "[2/5] Aggregate results (no case-level data)"
copy results/e04/formal_v1/protocol.json            results/e04/protocol.json
for f in "$SRC"/results/e04/formal_v1/final/*.json; do
  [ -f "$f" ] && copy "results/e04/formal_v1/final/$(basename "$f")" "results/e04/$(basename "$f")"
done
for f in protocol.json results.json supplementary_stats.json fairness_summary.json fairness_subgroups.csv; do
  copy "results/e05/formal_test/$f" "results/e05/formal_test/$f"
done
for f in oct_axis_decision.json domain_gap_findings.json e05_preflight.json; do
  copy "results/e05/audit/$f" "results/e05/audit/$f"
done
copy results/e05/e05b_summary.csv results/e05/e05b_summary.csv
for d in runs runs_d; do
  for m in "$SRC"/results/e05/$d/*/metrics.json; do
    [ -f "$m" ] || continue
    run="$(basename "$(dirname "$m")")"
    case "$run" in smoke_*) continue ;; esac
    copy "results/e05/$d/$run/metrics.json" "results/e05/run_metrics/$run.json"
  done
done

echo "[3/5] Exported environment"
if command -v conda >/dev/null 2>&1; then
  conda env export -n opticfusion --no-builds | grep -v '^prefix:' > environment.exported.yml && echo "  + environment.exported.yml"
else
  echo "  ! conda not found; skip export"
fi

echo "[4/5] Checksums"
mkdir -p docs
find . -path ./.git -prune -o -type f \( -name '*.py' -o -name '*.json' -o -name '*.csv' -o -name '*.sh' -o -name '*.yml' \) -print \
  | sort | xargs sha256sum > docs/checksums.sha256
python - <<'PY'
import json
p = json.load(open("results/e05/formal_test/protocol.json"))
with open("docs/locked_checkpoints.sha256", "w") as f:
    f.write(f"{p['script_sha256']}  e05/e05_formal_test.py (evaluation script, locked)\n")
    for name, h in sorted(p["checkpoint_sha256"].items()):
        f.write(f"{h}  results/e05/runs/{name}/best.pt (not distributed)\n")
print("  + docs/locked_checkpoints.sha256")
PY

echo "[5/5] Release check"
python tools/check_release.py --verify-lock
