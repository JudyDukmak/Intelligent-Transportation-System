"""
Merged baseline: flow preprocessing (from ``preprocessing/Baseline.py``) + toy Gym env + PPO
(from ``Baseline1.ipynb``). Run from the ``code/`` directory::

    python -m baseline.traffic_baseline preprocess
    python -m baseline.traffic_baseline train --timesteps 30000

CityFlow signal control (real sim) is a **separate** step: wrap ``cityflow.Engine`` with Gym,
use ``set_tl_phase`` + lane observations; this file keeps the **abstract** queue baseline
that matches your notebook.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOW_PATH = ROOT / "data" / "flow.json"
ARRIVALS_PATH = ROOT / "preprocessing" / "traffic_arrivals.json"


def load_flow_entries(path: Path | None = None) -> list[dict]:
    path = path or FLOW_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("flow.json must be a JSON array")
    return data


def build_traffic_counter(flow: list[dict], shift_to_zero: bool = True) -> Counter:
    """Vehicles entering at time t (same as notebook; optional shift so min time is 0)."""
    start_times = [int(v["startTime"]) for v in flow]
    if shift_to_zero and start_times:
        m = min(start_times)
        start_times = [t - m for t in start_times]
    return Counter(start_times)


def write_traffic_arrivals_json(
    out_path: Path | None = None,
    flow_path: Path | None = None,
    shift_to_zero: bool = True,
) -> Path:
    out_path = out_path or ARRIVALS_PATH
    flow = load_flow_entries(flow_path)
    traffic = build_traffic_counter(flow, shift_to_zero=shift_to_zero)
    t_max = max(traffic) if traffic else 0
    payload = {
        "n_vehicles": len(flow),
        "t_max": t_max,
        "shift_to_zero": shift_to_zero,
        "traffic": {str(t): traffic[t] for t in sorted(traffic)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  vehicles: {payload['n_vehicles']}, t_max: {t_max}, unique arrival times: {len(traffic)}")
    return out_path


def load_traffic_counter_from_json(path: Path | None = None) -> Counter:
    path = path or ARRIVALS_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    tr = data.get("traffic", {})
    return Counter({int(k): int(v) for k, v in tr.items()})


# --------------------------------------------------------------------------- gym env (notebook)
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None  # type: ignore
    spaces = None  # type: ignore


class ToyTrafficEnv(gym.Env if gym else object):  # type: ignore[misc, valid-type]
    """
    Queue toy from Baseline1.ipynb: 4 queues, 2 actions, reward -sum(queue).
    ``traffic`` maps simulation time -> number of vehicles entering in that second.
    """

    metadata = {"render_modes": []}

    def __init__(self, traffic: Counter, horizon: int = 3600, dt: int = 10):
        if gym is None:
            raise ImportError("Install gymnasium: pip install gymnasium")
        super().__init__()
        self.traffic = traffic
        self.horizon = horizon
        self.dt = dt
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        self.time = 0
        self.queue = np.zeros(4, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.queue = np.zeros(4, dtype=np.float32)
        return self.queue.copy(), {}

    def step(self, action):
        arrivals = sum(self.traffic.get(self.time + i, 0) for i in range(self.dt))
        if arrivals > 0:
            probs = [0.25, 0.25, 0.25, 0.25]
            incoming = self.np_random.multinomial(arrivals, probs)
            self.queue = self.queue + incoming.astype(np.float32)
        else:
            incoming = np.zeros(4, dtype=np.float32)

        if action == 0:
            self.queue[0] = max(0.0, self.queue[0] - 4)
            self.queue[1] = max(0.0, self.queue[1] - 4)
        elif action == 1:
            self.queue[2] = max(0.0, self.queue[2] - 4)
            self.queue[3] = max(0.0, self.queue[3] - 4)

        reward = float(-np.sum(self.queue))
        self.time += self.dt
        terminated = self.time > self.horizon
        return self.queue.copy(), reward, terminated, False, {}


def train_ppo(timesteps: int = 30_000, traffic: Counter | None = None) -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise ImportError("pip install stable-baselines3 torch") from e

    if traffic is None:
        traffic = load_traffic_counter_from_json()
    env = ToyTrafficEnv(traffic)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
    )
    model.learn(total_timesteps=timesteps)
    out = ROOT / "preprocessing" / "toy_ppo_model.zip"
    model.save(str(out.with_suffix("")))
    print(f"Saved model to {out.with_suffix('')}.zip")


def main_preprocess() -> None:
    write_traffic_arrivals_json()


def main_train(args: argparse.Namespace) -> None:
    write_traffic_arrivals_json()
    train_ppo(timesteps=args.timesteps)


def main() -> None:
    p = argparse.ArgumentParser(description="Traffic baseline: preprocess flow + optional toy PPO")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preprocess", help="Write preprocessing/traffic_arrivals.json from data/flow.json")
    p_pre.set_defaults(func=lambda _: main_preprocess())

    p_tr = sub.add_parser("train", help="Preprocess + train PPO on ToyTrafficEnv (needs gymnasium, SB3)")
    p_tr.add_argument("--timesteps", type=int, default=30_000)
    p_tr.set_defaults(func=main_train)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
