import json

with open("data/roadnet.json") as f:
    roadnet = json.load(f)

for inter in roadnet["intersections"]:
    if inter["id"] == "intersection_1_1":
        rl = inter["roadLinks"]

        for pid, ph in enumerate(inter["trafficLight"]["lightphases"]):
            print(f"\nPHASE {pid}")

            for idx in ph["availableRoadLinks"]:
                r = rl[idx]
                print(
                    r["startRoad"],
                    "->",
                    r["endRoad"],
                    "|",
                    r["type"]
                )