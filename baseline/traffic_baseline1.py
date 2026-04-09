import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import gym
from gym import spaces

import cityflow

# =========================
# PATHS
# =========================
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.json"
ROADNET_PATH = ROOT / "data" / "roadnet.json"

TARGET_INTERSECTION = "intersection_1_1"


# =========================
# ENVIRONMENT
# =========================
class CityFlowEnv(gym.Env):

    def __init__(self):
        super(CityFlowEnv, self).__init__()

        # init engine
        self.eng = cityflow.Engine(str(CONFIG_PATH), thread_num=1)

        # load roadnet
        self.roadnet = json.loads(ROADNET_PATH.read_text())

        # get intersection
        self.intersection = None
        for inter in self.roadnet["intersections"]:
            if inter["id"] == TARGET_INTERSECTION:
                self.intersection = inter
                break

        if self.intersection is None:
            raise Exception("Intersection not found")

        # phases
        self.phases = self.intersection["trafficLight"]["lightphases"]
        self.n_phases = len(self.phases)

        # action space = choose phase
        self.action_space = spaces.Discrete(self.n_phases)

        # get incoming lanes
        self.in_lanes = self._get_incoming_lanes()

        # observation space
        obs_dim = 2 + len(self.in_lanes) * 3
        self.observation_space = spaces.Box(low=0, high=1000, shape=(obs_dim,), dtype=np.float32)

        self.current_phase = 0
        self.phase_time = 0

    # =========================
    # ROAD STRUCTURE
    # =========================
    def _get_incoming_lanes(self):
        lanes = set()
        for rl in self.intersection["roadLinks"]:
            start = rl["startRoad"]
            for ll in rl["laneLinks"]:
                lane_id = "%s_%d" % (start, ll["startLaneIndex"])
                lanes.add(lane_id)
        return sorted(list(lanes))

    def _get_outgoing_map(self):
        mapping = {}
        for rl in self.intersection["roadLinks"]:
            start = rl["startRoad"]
            end = rl["endRoad"]

            for ll in rl["laneLinks"]:
                in_lane = "%s_%d" % (start, ll["startLaneIndex"])
                out_lane = "%s_%d" % (end, ll["endLaneIndex"])

                if in_lane not in mapping:
                    mapping[in_lane] = []
                mapping[in_lane].append(out_lane)
        return mapping

    # =========================
    # STATE
    # =========================
    def _get_state(self):
        lane_counts = self.eng.get_lane_vehicle_count()
        waiting = self.eng.get_lane_waiting_vehicle_count()

        outgoing_map = self._get_outgoing_map()

        state = []

        # phase info
        state.append(self.current_phase)
        state.append(self.phase_time)

        for lane in self.in_lanes:
            q = lane_counts.get(lane, 0)
            w = waiting.get(lane, 0)

            # outgoing vehicles (REAL)
            out = 0
            for out_lane in outgoing_map.get(lane, []):
                out += lane_counts.get(out_lane, 0)

            state.extend([q, w, out])

        return np.array(state, dtype=np.float32)

    # =========================
    # REWARD (MAX PRESSURE)
    # =========================
    def _compute_reward(self):
        lane_counts = self.eng.get_lane_vehicle_count()
        outgoing_map = self._get_outgoing_map()

        pressure = 0

        for lane in self.in_lanes:
            q_in = lane_counts.get(lane, 0)

            q_out = 0
            for out_lane in outgoing_map.get(lane, []):
                q_out += lane_counts.get(out_lane, 0)

            pressure += (q_in - q_out)

        return -pressure  # minimize pressure

    # =========================
    # STEP
    # =========================
    def step(self, action):

        if action != self.current_phase:
            self.eng.set_tl_phase(TARGET_INTERSECTION, int(action))
            self.current_phase = action
            self.phase_time = 0

        # simulate
        for _ in range(10):
            self.eng.next_step()
            self.phase_time += 1

        state = self._get_state()
        reward = self._compute_reward()

        done = False

        return state, reward, done, {}

    # =========================
    # RESET
    # =========================
    def reset(self):
        self.eng = cityflow.Engine(str(CONFIG_PATH), thread_num=1)
        self.current_phase = 0
        self.phase_time = 0
        return self._get_state()

    def render(self):
        pass


# =========================
# TRAIN PPO
# =========================
def train():

    from stable_baselines3 import PPO

    env = CityFlowEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
    )

    model.learn(total_timesteps=30000)

    model.save(str(ROOT / "results" / "ppo_traffic"))

    print("Model saved.")

 # =========================================================
 # EVALUATION
 # =========================================================
def evaluate():
    from stable_baselines3 import PPO

    env = CityFlowEnv()
    model = PPO.load(str(ROOT / "results" / "ppo_traffic"))

    state = env.reset()

    total_wait = []
    total_queue = []
    throughput = []

    max_steps = 500  # prevent infinite loop

    for step in range(max_steps):
        action, _ = model.predict(state)
        state, _, _, _ = env.step(action)

        wait = sum(env.eng.get_lane_waiting_vehicle_count().values())
        queue = sum(env.eng.get_lane_vehicle_count().values())
        vehicles = env.eng.get_vehicle_count()

        total_wait.append(wait)
        total_queue.append(queue)
        throughput.append(vehicles)

    print("\n=== METRICS ===")
    print("AWT (Avg Waiting Time):", np.mean(total_wait))
    print("AQL (Avg Queue Length):", np.mean(total_queue))
    print("ATT (Avg Travel Time):", env.eng.get_average_travel_time())
    print("Throughput:", throughput[-1])

    # Save results for plotting
    np.save(ROOT / "results" / "wait.npy", total_wait)
    np.save(ROOT / "results" / "queue.npy", total_queue)
    np.save(ROOT / "results" / "throughput.npy", throughput)

    print("Saved raw metrics to results/")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)

    args = parser.parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "eval":
        evaluate()


if __name__ == "__main__":
    main()
    