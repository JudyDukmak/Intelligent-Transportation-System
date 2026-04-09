import numpy as np
from env.cityflow_env import CityFlowEnv
from stable_baselines3 import PPO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "baseline2"

def evaluate():
    env = CityFlowEnv()
    model = PPO.load(str(RESULT_PATH / "ppo_traffic"))

    state = env.reset()
    total_wait, total_queue, throughput = [], [], []

    max_steps = 3600
    for _ in range(max_steps):
        action, _ = model.predict(state)
        state, _, done, _ = env.step(action)
        total_wait.append(sum(env.eng.get_lane_waiting_vehicle_count().values()))
        total_queue.append(sum(env.eng.get_lane_vehicle_count().values()))
        throughput.append(env.eng.get_vehicle_count())
        if done:
            break

    print("\n=== METRICS ===")
    print("AWT:", np.mean(total_wait))
    print("AQL:", np.mean(total_queue))
    print("ATT:", env.eng.get_average_travel_time())
    print("Throughput:", throughput[-1])

    # Save raw results
    np.save(RESULT_PATH / "wait.npy", total_wait)
    np.save(RESULT_PATH / "queue.npy", total_queue)
    np.save(RESULT_PATH / "throughput.npy", throughput)
    print("Saved results for plotting.")