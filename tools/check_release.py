#!/usr/bin/env python
"""Check the repository before publishing.

Fails if it finds licensed images, model weights, prediction arrays, large files, or case-level identifiers
in CSV/JSON files. With --verify-lock, also confirms that e05/e05_formal_test.py matches the SHA-256
recorded in results/e05/formal_test/protocol.json.
"""
import argparse, csv, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_EXT = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".zip", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm", ".nii", ".gz"}
ID_KEYS = {"case_id", "patient_id", "npz_path", "jpg_path", "filename", "fundus_path", "oct_path"}
MAX_MB = 10
CASE_ID = re.compile(r"^\d{4,5}$")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def json_issues(obj, path=""):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in ID_KEYS:
                found.append(f"key '{k}' at {path or '/'}")
            found += json_issues(v, f"{path}/{k}")
    elif isinstance(obj, list):
        ids = [x for x in obj if isinstance(x, str) and CASE_ID.match(x)]
        if len(ids) >= 20:
            found.append(f"list of {len(ids)} case-like identifiers at {path or '/'}")
        for i, v in enumerate(obj[:2000]):
            found += json_issues(v, f"{path}[{i}]")
    return found

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify-lock", action="store_true"); a = ap.parse_args()
    issues = []
    for p in sorted(ROOT.rglob("*")):
        if ".git" in p.parts or not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if rel.parts and rel.parts[0] == "data":
            issues.append(f"{rel}: data directory must not be published"); continue
        if p.suffix.lower() in FORBIDDEN_EXT:
            issues.append(f"{rel}: forbidden file type ({p.suffix})")
        if p.stat().st_size > MAX_MB * 1024 * 1024:
            issues.append(f"{rel}: larger than {MAX_MB} MB")
        if p.suffix.lower() == ".csv":
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                header = next(csv.reader(f), [])
            bad = [h for h in header if h.strip().lower() in ID_KEYS]
            if bad:
                issues.append(f"{rel}: case-level column(s) {bad}")
        if p.suffix.lower() == ".json":
            try:
                found = json_issues(json.loads(p.read_text(encoding="utf-8")))
            except Exception as e:
                found = [f"unreadable JSON ({e})"]
            issues += [f"{rel}: {x}" for x in found[:5]]

    if a.verify_lock:
        proto = ROOT / "results/e05/formal_test/protocol.json"
        script = ROOT / "e05/e05_formal_test.py"
        if not proto.exists() or not script.exists():
            issues.append("lock verification: protocol.json or e05/e05_formal_test.py missing")
        else:
            want = json.loads(proto.read_text())["script_sha256"]; got = sha(script)
            if want == got:
                print(f"LOCK VERIFIED: e05/e05_formal_test.py sha256={got[:16]}... matches protocol.json")
            else:
                issues.append(f"lock verification FAILED: script sha256 {got[:16]}... != locked {want[:16]}...")

    if issues:
        print("RELEASE CHECK FAILED:")
        for x in issues: print("  -", x)
        sys.exit(1)
    print("RELEASE CHECK PASSED: no licensed data, weights, predictions or case identifiers found")

if __name__ == "__main__":
    main()
