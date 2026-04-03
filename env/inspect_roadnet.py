import json
from pathlib import Path

import cityflow


ROOT = Path(__file__).resolve().parents[1]  # .../code
ROADNET_PATH = ROOT / "data" / "roadnet.json"
CONFIG_PATH = "config/config.json"  # relative to /work when run in docker with -w /work


def safe_get(d, key, default=None):
    v = d.get(key, default)
    return default if v is None else v


def lane_ids_from_lane_dict_api(eng: cityflow.Engine):
    """
    Official API: get_lane_vehicle_count() takes NO lane_id argument.
    It returns dict {lane_id: count} for all lanes (see CityFlow docs Quick Start).
    """
    if not hasattr(eng, "get_lane_vehicle_count"):
        return {}
    d = eng.get_lane_vehicle_count()
    return d if isinstance(d, dict) else {}


def lane_ids_from_vehicles(eng: cityflow.Engine, steps: int = 200):
    """
    Docs: get_vehicle_info keys include 'drivable' (current lane or lanelink id), all str.
    """
    if not (hasattr(eng, "get_vehicles") and hasattr(eng, "get_vehicle_info") and hasattr(eng, "next_step")):
        return []
    for _ in range(steps):
        eng.next_step()
    vids = eng.get_vehicles()
    drivables = set()
    for vid in vids[:300]:
        try:
            info = eng.get_vehicle_info(vid)
        except Exception:
            continue
        if not isinstance(info, dict):
            continue
        # Official key is 'drivable' (lane or lanelink)
        d = info.get("drivable") or info.get("lane")
        if isinstance(d, str) and d:
            drivables.add(d)
    return sorted(drivables)


def main():
    if not ROADNET_PATH.exists():
        raise FileNotFoundError(f"Missing roadnet: {ROADNET_PATH}")

    roadnet = json.loads(ROADNET_PATH.read_text(encoding="utf-8"))

    intersections = roadnet.get("intersections", [])
    roads = roadnet.get("roads", [])

    print(f"Loaded roadnet: {ROADNET_PATH}")
    print(f"Intersections: {len(intersections)}")
    print(f"Roads: {len(roads)}")

    print("\n=== Intersections + traffic light phases ===")
    for it in intersections:
        iid = it.get("id")
        virtual = it.get("virtual", False)
        tl = safe_get(it, "trafficLight", {}) or {}
        phases = tl.get("lightphases", []) or []
        print(f"\n- intersection_id={iid} virtual={virtual} phases={len(phases)}")
        for i, ph in enumerate(phases):
            t = ph.get("time")
            avail = ph.get("availableRoadLinks", [])
            print(f"  phase_id={i} time={t} availableRoadLinks={len(avail)}")

    eng = cityflow.Engine(CONFIG_PATH, thread_num=1)
    print("\nEngine ready.")

    print("\n=== Note: snapshot() ===")
    print("eng.snapshot() returns an Archive object (save/load), not lane JSON. Ignore for lane listing.")
    eng.snapshot()
    print("Snapshot taken.")
    print("\n=== Lane ids from get_lane_vehicle_count()  ===")
    lane_ids = []
    total_steps = 0
    for target_steps in (0, 50, 250):
        while total_steps < target_steps:
            eng.next_step()
            total_steps += 1
        lane_counts = lane_ids_from_lane_dict_api(eng)
        n_veh = eng.get_vehicle_count() if hasattr(eng, "get_vehicle_count") else -1
        print(f"After {total_steps} steps: vehicles={n_veh} Total_lanes_in_dict={len(lane_counts)}")
        if lane_counts:
            lane_ids = sorted(lane_counts.keys())
            print("Sample lane ids:", lane_ids[:20])
            for k in list(lane_counts.keys())[:5]:
                print(f"  {k!r} -> {lane_counts[k]}")
            break

    if not lane_ids:
        print("\n=== Drivable ids from get_vehicle_info()['drivable'] (after more steps) ===")
        lane_ids = lane_ids_from_vehicles(eng, steps=300)
        if lane_ids:
            print(f"Total drivable ids seen: {len(lane_ids)}")
            print("Sample:", lane_ids[:20])
        else:
            print("Still empty — check config paths, flow startTime, and that rlTrafficLight allows sim to run.")

    print("\n=== Road -> lane ids (prefix match on discovered ids) ===")
    for r in roads:
        rid = r.get("id")
        n = len(r.get("lanes", []) or [])
        matches = []
        if lane_ids:
            matches = [lid for lid in lane_ids if isinstance(lid, str) and lid.startswith(str(rid))]
        print(f"- road_id={rid} lanes_in_roadnet={n} matched_ids={len(matches)} sample={matches[:8]}")


if __name__ == "__main__":
    main()

