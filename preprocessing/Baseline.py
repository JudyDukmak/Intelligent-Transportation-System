"""
Preprocess CityFlow flow.json into an arrival-time histogram (same idea as Baseline1.ipynb).

Reads:  code/data/flow.json
Writes: code/preprocessing/traffic_arrivals.json

The notebook uses ``traffic = Counter(start_times)`` so that ``traffic[t]`` is how many
vehicles are scheduled to *enter* at simulation time t. That drives the toy TrafficEnv.

Run locally or in Docker from ``code/``:
  python preprocessing/Baseline.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOW_PATH = ROOT / "data" / "flow.json"
OUT_PATH = ROOT / "preprocessing" / "traffic_arrivals.json"


def load_flow(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("flow.json must be a JSON array of vehicle flow entries")
    return data


def main() -> None:
    flow = load_flow(FLOW_PATH)
    start_times = [entry["startTime"] for entry in flow]
    traffic = Counter(start_times)
    t_max = max(start_times) if start_times else 0

    payload = {
        "n_vehicles": len(flow),
        "t_max": t_max,
        "traffic": {str(t): traffic[t] for t in sorted(traffic)},
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(f"  vehicles in flow: {payload['n_vehicles']}")
    print(f"  max startTime: {t_max}")
    print(f"  distinct start times with arrivals: {len(traffic)}")


if __name__ == "__main__":
    main()
