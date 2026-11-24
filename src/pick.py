"""Helpers for the experiment pipeline.
  python src/pick.py best      -> best phase-B encoder (by CV window accuracy)
  python src/pick.py ensemble  -> top-5 IMU+FP+EMG runs (same window list) for soft voting
"""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
runs = {}
for line in open(RES / "results.jsonl"):
    r = json.loads(line)
    runs[r["name"]] = r

if sys.argv[1] == "best":
    b = max((r for n, r in runs.items() if n.startswith("B_")), key=lambda r: r["window_acc"])
    print(b["encoder"])
else:
    pool = [r for r in runs.values() if r["mods"] == ["IMU", "FP", "EMG"] and not r.get("gan_only")]
    print(" ".join(r["name"] for r in sorted(pool, key=lambda r: -r["window_acc"])[:5]))
