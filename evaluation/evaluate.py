import numpy as np
from env.cityflow_env import CityFlowEnv
from stable_baselines3 import PPO
from pathlib import Path
from config.load_config import load_config

cfg = load_config()

BASELINE_NAME = cfg["general"]["baseline_name"]

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / BASELINE_NAME


def evaluate():

    env = CityFlowEnv()
    model = PPO.load(str(RESULT_PATH / "ppo_traffic"))

    state = env.reset()

    total_wait = []
    total_queue = []

    throughput_per_step = []
    throughput_total = []

    max_steps = 3600

    # -------------------------------
    # throughput tracking (FIXED)
    # -------------------------------
    prev_total_vehicles = env.eng.get_vehicle_count()

    for step in range(max_steps):

        action, _ = model.predict(state)
        state, _, done, _ = env.step(action)

        # ---------------- WAIT ----------------
        wait = sum(env.eng.get_lane_waiting_vehicle_count().values())

        # ---------------- QUEUE ----------------
        queue = sum(
            env.eng.get_lane_vehicle_count().get(lane, 0)
            for lane in env.in_lanes
        )

        # ---------------- THROUGHPUT (FIXED) ----------------
        current_total = env.eng.get_vehicle_count()

        delta_throughput = prev_total_vehicles - current_total
        delta_throughput = max(0, delta_throughput)

        throughput_per_step.append(delta_throughput)
        throughput_total.append(delta_throughput)

        prev_total_vehicles = current_total

        # ---------------- STORE METRICS ----------------
        total_wait.append(wait)
        total_queue.append(queue)

        if done:
            break

    # ================= FINAL METRICS =================
    print("\n=== METRICS ===")

    print("AWT (Avg Wait Time):", np.mean(total_wait))
    print("AQL (Avg Queue Length):", np.mean(total_queue))
    print("ATT (Avg Travel Time):", env.eng.get_average_travel_time())

    print("Total Throughput:", np.sum(throughput_total))

    # ================= SAVE =================
    np.save(RESULT_PATH / "wait.npy", total_wait)
    np.save(RESULT_PATH / "queue.npy", total_queue)
    np.save(RESULT_PATH / "throughput.npy", throughput_total)


    print("Saved results to:", RESULT_PATH)

if __name__ == "__main__": evaluate()