"""
CityFlow roadnet / flow JSON helpers.

Lane ids used by the Engine match ``{road_id}_{lane_index}`` on each road (see
``get_lane_vehicle_count()`` keys), not ``{road}_{rl_index}_{lane}``.

``roadLinks[].type`` values are CityFlow strings, e.g. ``go_straight``, ``turn_left``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

# Default paths relative to the ``code/`` package root
_CODE_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path | str) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def load_roadnet(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else _CODE_ROOT / "data" / "roadnet.json"
    data = load_json(p)
    if not isinstance(data, dict):
        raise ValueError("roadnet must be a JSON object")
    return data


def load_flow(path: Path | str | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path is not None else _CODE_ROOT / "data" / "flow.json"
    data = load_json(p)
    if not isinstance(data, list):
        raise ValueError("flow.json must be a JSON array")
    return data


def get_intersection(roadnet: dict[str, Any], intersection_id: str) -> Optional[dict[str, Any]]:
    for inter in roadnet.get("intersections", []):
        if inter.get("id") == intersection_id:
            return inter
    return None


def parse_roadlinks(intersection: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a flat list of road-link records (start/end/type)."""
    out = []
    for rl in intersection.get("roadLinks", []):
        out.append(
            {
                "startRoad": rl["startRoad"],
                "endRoad": rl["endRoad"],
                "type": rl.get("type", "unknown"),
            }
        )
    return out


def build_lane_mapping(roadnet: dict[str, Any], intersection_id: str) -> dict[str, str]:
    """
    Map each **incoming** approach lane id at this intersection to a movement ``type``.

    CityFlow lane id for a lane on ``startRoad`` is ``f"{startRoad}_{startLaneIndex}"``
    (same convention as ``Engine.get_lane_vehicle_count()`` keys).
    """
    intersection = get_intersection(roadnet, intersection_id)
    if intersection is None:
        raise KeyError(f"intersection not found: {intersection_id!r}")

    lane_map: dict[str, str] = {}
    for rl in intersection.get("roadLinks", []):
        movement = rl.get("type", "unknown")
        for ll in rl.get("laneLinks", []):
            start_lane = ll["startLaneIndex"]
            lane_id = f"{rl['startRoad']}_{start_lane}"
            lane_map[lane_id] = movement
    return lane_map


def transform_state(
    raw_lane_counts: dict[str, int],
    lane_map: dict[str, str],
) -> dict[str, int]:
    """Aggregate per-lane counts into totals per movement type (e.g. go_straight, turn_left)."""
    agg: Counter[str] = Counter()
    for lane_id, count in raw_lane_counts.items():
        m = lane_map.get(lane_id)
        if m is not None:
            agg[m] += count
    return dict(agg)


def analyze_flow(flow: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize flow.json for inspection (no CityFlow required)."""
    routes = [tuple(f.get("route", [])) for f in flow]
    route_counts = Counter(routes)
    return {
        "n_vehicles": len(flow),
        "unique_routes": len(route_counts),
        "top_routes": route_counts.most_common(10),
    }


class RoadnetParser:
    """OOP wrapper: load once, query many intersections."""

    def __init__(self, roadnet_path: Path | str | None = None):
        self.path = Path(roadnet_path) if roadnet_path else _CODE_ROOT / "data" / "roadnet.json"
        self.roadnet = load_roadnet(self.path)
        self._by_id = {i["id"]: i for i in self.roadnet.get("intersections", []) if "id" in i}

    def intersection(self, intersection_id: str) -> dict[str, Any]:
        if intersection_id not in self._by_id:
            raise KeyError(intersection_id)
        return self._by_id[intersection_id]

    def lane_movement_map(self, intersection_id: str) -> dict[str, str]:
        return build_lane_mapping(self.roadnet, intersection_id)


def _demo() -> None:
    rn = load_roadnet()
    iid = "intersection_1_1"
    m = build_lane_mapping(rn, iid)
    print(f"intersection {iid}: {len(m)} approach lanes mapped")
    for k in sorted(m.keys())[:8]:
        print(f"  {k} -> {m[k]}")
    fl = load_flow()
    print("flow summary:", analyze_flow(fl))


if __name__ == "__main__":
    _demo()
