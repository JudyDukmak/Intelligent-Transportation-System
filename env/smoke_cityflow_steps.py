"""
Quick sanity check: step the CityFlow engine and watch vehicles / one lane.

Run from repo `code/` in Docker:
  python env/smoke_cityflow_steps.py
"""
import cityflow

CONFIG_PATH = "config/config.json"
TOTAL_STEPS = 500
LOG_EVERY = 50
# Any lane that appears on routes in flow.json, e.g. road_1_2_3 -> road_1_2_3_0
SAMPLE_LANE = "road_1_2_3_0"


def main():
    eng = cityflow.Engine(CONFIG_PATH, thread_num=1)
    for t in range(TOTAL_STEPS):
        eng.next_step()
        if t % LOG_EVERY == 0 or t == TOTAL_STEPS - 1:
            counts = eng.get_lane_vehicle_count()
            print(
                f"step={t}  sim_time={eng.get_current_time():.1f}  "
                f"vehicles={eng.get_vehicle_count()}  "
                f"lane[{SAMPLE_LANE}]={counts.get(SAMPLE_LANE, 'n/a')}"
            )


if __name__ == "__main__":
    main()
