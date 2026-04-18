import numpy as np
import json
from pathlib import Path
import gym
from gym import spaces
import cityflow

# =====================================================
# PATHS
# =====================================================
ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "config" / "config.json"
ROADNET_PATH = ROOT / "data" / "roadnet.json"
HYPERPARAMS_PATH = ROOT / "config" / "hyperparams.json"

TARGET_INTERSECTION = "intersection_1_1"

# =====================================================
# LOAD HYPERPARAMETERS
# =====================================================
with open(HYPERPARAMS_PATH, "r") as f:
    HP = json.load(f)

MAX_QUEUE = HP["env"]["max_queue"]
CONTROL_INTERVAL = HP["env"]["control_interval"]
MIN_GREEN = HP["env"]["min_green"]
YELLOW_TIME = HP["env"]["yellow_time"]
MAX_EPISODE_STEPS = HP["env"]["max_episode_steps"]

# reward weights
W_PRESSURE = HP["reward"]["pressure"]
W_QUEUE = HP["reward"]["queue"]
W_WAIT = HP["reward"]["wait"]
W_THROUGHPUT = HP["reward"]["throughput"]
W_SWITCH = HP["reward"]["switch"]

# =====================================================
# ENVIRONMENT
# =====================================================
class CityFlowEnv(gym.Env):

    def __init__(self):
        super(CityFlowEnv, self).__init__()

        # ---------------- ENGINE ----------------
        self.eng = cityflow.Engine(str(CONFIG_PATH), thread_num=1)

        # ---------------- LOAD ROADNET ----------------
        self.roadnet = json.loads(ROADNET_PATH.read_text())

        self.intersection = None
        for inter in self.roadnet["intersections"]:
            if inter["id"] == TARGET_INTERSECTION:
                self.intersection = inter
                break

        if self.intersection is None:
            raise Exception("Intersection not found")

        # ---------------- PHASES ----------------
        self.phases = self.intersection["trafficLight"]["lightphases"]
        self.n_phases = len(self.phases)

        self.action_space = spaces.Discrete(self.n_phases)

        # ---------------- LANES ----------------
        self.in_lanes = self._get_incoming_lanes()

        # ---------------- CACHE STATIC STRUCTURES ----------------
        self.outgoing_map = self._get_outgoing_map()
        self.phase_lane_map = self._build_phase_lane_map()

        # ---------------- OBSERVATION SPACE ----------------
        # phase one-hot + phase_time + lane data(q,w,out,delta_q)
        obs_dim = self.n_phases + 1 + len(self.in_lanes) * 4

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # ---------------- STATE ----------------
        self.current_phase = 0
        self.phase_time = 0
        self.elapsed_steps = 0
        self.max_episode_steps = MAX_EPISODE_STEPS

        # ---------------- MEMORY ----------------
        self.prev_total_exited = 0
        self.prev_lane_counts = {lane: 0 for lane in self.in_lanes}
        self.prev_total_vehicles = 0

    # =====================================================
    # SEED
    # =====================================================
    def seed(self, seed=None):
        np.random.seed(seed)

    # =====================================================
    # ROAD STRUCTURE
    # =====================================================
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

    def _build_phase_lane_map(self):
        phase_map = {}

        for phase_id, phase in enumerate(self.phases):
            lanes = []

            for rl_idx in phase["availableRoadLinks"]:
                rl = self.intersection["roadLinks"][rl_idx]

                for ll in rl["laneLinks"]:
                    lane = "%s_%d" % (
                        rl["startRoad"],
                        ll["startLaneIndex"]
                    )
                    lanes.append(lane)

            phase_map[phase_id] = lanes

        return phase_map

    # =====================================================
    # VALID ACTIONS
    # =====================================================
    def _get_valid_actions(self):
        lane_counts = self.eng.get_lane_vehicle_count()

        valid = []

        for phase_id, lanes in self.phase_lane_map.items():
            score = sum(lane_counts.get(l, 0) for l in lanes)

            if score > 0:
                valid.append(phase_id)

        if len(valid) == 0:
            valid = [self.current_phase]

        return valid

    # =====================================================
    # STATE
    # =====================================================
    def _get_state(self):
        lane_counts = self.eng.get_lane_vehicle_count()
        waiting = self.eng.get_lane_waiting_vehicle_count()

        state = []

        # -----------------------------------
        # FIX 1: One-hot encode phase
        # -----------------------------------
        phase_vec = np.zeros(self.n_phases, dtype=np.float32)
        phase_vec[self.current_phase] = 1.0
        state.extend(phase_vec)

        # -----------------------------------
        # FIX 2: phase_time clipped
        # -----------------------------------
        phase_time_norm = min(self.phase_time / float(MIN_GREEN), 1.0)
        state.append(phase_time_norm)

        # -----------------------------------
        # lane features
        # q, w, out, delta_q
        # -----------------------------------
        for lane in self.in_lanes:

            q = lane_counts.get(lane, 0) / MAX_QUEUE
            w = waiting.get(lane, 0) / MAX_QUEUE

            out = sum(
                lane_counts.get(l, 0)
                for l in self.outgoing_map.get(lane, [])
            ) / MAX_QUEUE

            # FIX 3: delta queue
            prev_q = self.prev_lane_counts.get(lane, 0)
            current_q = lane_counts.get(lane, 0)

            delta_q = (current_q - prev_q) / MAX_QUEUE
            delta_q = np.clip(delta_q, -1.0, 1.0)

            state.extend([q, w, out, delta_q])

        # update memory
        self.prev_lane_counts = {
            lane: lane_counts.get(lane, 0)
            for lane in self.in_lanes
        }

        return np.array(state, dtype=np.float32)

    # =====================================================
    # REWARD
    # =====================================================
    def _compute_reward(self, action_changed):

        lane_counts = self.eng.get_lane_vehicle_count()
        waiting = self.eng.get_lane_waiting_vehicle_count()

        # -----------------------------------
        # PRESSURE
        # -----------------------------------
        pressure = 0

        for lane in self.in_lanes:
            q_in = lane_counts.get(lane, 0)

            q_out = sum(
                lane_counts.get(l, 0)
                for l in self.outgoing_map.get(lane, [])
            )

            pressure += (q_in - q_out)

        pressure /= (len(self.in_lanes) * MAX_QUEUE)

        # -----------------------------------
        # QUEUE ONLY INCOMING LANES
        # -----------------------------------
        queue_penalty = sum(
            lane_counts.get(lane, 0)
            for lane in self.in_lanes
        ) / (len(self.in_lanes) * MAX_QUEUE)

        # -----------------------------------
        # WAIT ONLY INCOMING LANES
        # -----------------------------------
        wait_penalty = sum(
            waiting.get(lane, 0)
            for lane in self.in_lanes
        ) / (len(self.in_lanes) * MAX_QUEUE)

        # -----------------------------------
        # THROUGHPUT REWARD
        # -----------------------------------
        current_total = self.eng.get_vehicle_count()

        throughput_gain = self.prev_total_vehicles - current_total
        throughput_gain = max(0, throughput_gain)

        self.prev_total_vehicles = current_total

        # -----------------------------------
        # SWITCH PENALTY
        # -----------------------------------
        switch_penalty = 1 if action_changed else 0

        # -----------------------------------
        # FINAL REWARD
        # -----------------------------------
        reward = (
            -W_PRESSURE * pressure
            -W_QUEUE * queue_penalty
            -W_WAIT * wait_penalty
            +W_THROUGHPUT * throughput_gain
            -W_SWITCH * switch_penalty
        )

        reward = np.clip(reward, -1.0, 1.0)

        return reward

    # =====================================================
    # STEP
    # =====================================================
    def step(self, action):

        # safe action
        action = int(action)
        action = max(0, min(action, self.n_phases - 1))

        valid_actions = self._get_valid_actions()

        if action not in valid_actions:
            action = self.current_phase

        action_changed = False

        # -----------------------------------
        # CHANGE PHASE WITH YELLOW
        # -----------------------------------
        if action != self.current_phase and self.phase_time >= MIN_GREEN:

            for _ in range(YELLOW_TIME):
                self.eng.next_step()
                self.elapsed_steps += 1

            self.eng.set_tl_phase(
                TARGET_INTERSECTION,
                action
            )

            self.current_phase = action
            self.phase_time = 0
            action_changed = True

        # -----------------------------------
        # NORMAL CONTROL INTERVAL
        # -----------------------------------
        for _ in range(CONTROL_INTERVAL):
            self.eng.next_step()

            self.phase_time += 1
            self.elapsed_steps += 1

        state = self._get_state()
        reward = self._compute_reward(action_changed)

        done = self.elapsed_steps >= self.max_episode_steps

        return state, reward, done, {}

    # =====================================================
    # RESET
    # =====================================================
    def reset(self):

        self.eng.reset()

        self.current_phase = 0
        self.phase_time = 0
        self.elapsed_steps = 0

        self.prev_total_exited = 0
        self.prev_lane_counts = {
            lane: 0 for lane in self.in_lanes
        }
        self.prev_total_vehicles = 0
        
        return self._get_state()

    # =====================================================
    # RENDER
    # =====================================================
    def render(self):
        pass