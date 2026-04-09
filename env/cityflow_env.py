import numpy as np
import json
from pathlib import Path
import gym
from gym import spaces
import cityflow

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.json"
ROADNET_PATH = ROOT / "data" / "roadnet.json"
TARGET_INTERSECTION = "intersection_1_1"
MAX_QUEUE = 50.0  # for normalization

class CityFlowEnv(gym.Env):

    def __init__(self):
        super(CityFlowEnv, self).__init__()

        # engine init
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

        # traffic light info
        self.phases = self.intersection["trafficLight"]["lightphases"]
        self.n_phases = len(self.phases)

        # action space
        self.action_space = spaces.Discrete(self.n_phases)

        # incoming lanes
        self.in_lanes = self._get_incoming_lanes()

        # observation space: normalized
        obs_dim = 2 + len(self.in_lanes) * 3
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.current_phase = 0
        self.phase_time = 0
        self.max_episode_steps = 3600  # 1 hour
        self.elapsed_steps = 0

    # ---------------------
    # ROAD STRUCTURE
    # ---------------------
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

    # ---------------------
    # STATE
    # ---------------------
    def _get_state(self):
        lane_counts = self.eng.get_lane_vehicle_count()
        waiting = self.eng.get_lane_waiting_vehicle_count()
        outgoing_map = self._get_outgoing_map()

        state = []
        # phase info
        state.append(float(self.current_phase) / self.n_phases)
        state.append(float(self.phase_time) / 50.0)  # normalize phase time

        for lane in self.in_lanes:
            q = lane_counts.get(lane, 0) / MAX_QUEUE
            w = waiting.get(lane, 0) / MAX_QUEUE
            out = sum(lane_counts.get(l, 0) for l in outgoing_map.get(lane, [])) / MAX_QUEUE
            state.extend([q, w, out])

        return np.array(state, dtype=np.float32)

    # ---------------------
    # REWARD
    # ---------------------
    def _compute_reward(self):
        lane_counts = self.eng.get_lane_vehicle_count()
        waiting = self.eng.get_lane_waiting_vehicle_count()
        outgoing_map = self._get_outgoing_map()

        pressure = 0
        for lane in self.in_lanes:
            q_in = lane_counts.get(lane, 0)
            q_out = sum(lane_counts.get(l, 0) for l in outgoing_map.get(lane, []))
            pressure += q_in - q_out

        # penalty for queue and waiting
        queue_penalty = sum(lane_counts.values())
        wait_penalty = sum(waiting.values())

        reward = -pressure - 0.1 * queue_penalty - 0.05 * wait_penalty
        return reward

    # ---------------------
    # STEP
    # ---------------------
    def step(self, action):

        # minimum green time
        if action != self.current_phase and self.phase_time >= 10:
            self.eng.set_tl_phase(TARGET_INTERSECTION, int(action))
            self.current_phase = action
            self.phase_time = 0

        for _ in range(10):
            self.eng.next_step()
            self.phase_time += 1
            self.elapsed_steps += 1

        state = self._get_state()
        reward = self._compute_reward()
        done = self.elapsed_steps >= self.max_episode_steps

        return state, reward, done, {}

    # ---------------------
    # RESET
    # ---------------------
    def reset(self):
        self.eng = cityflow.Engine(str(CONFIG_PATH), thread_num=1)
        self.current_phase = 0
        self.phase_time = 0
        self.elapsed_steps = 0
        return self._get_state()

    def render(self):
        pass