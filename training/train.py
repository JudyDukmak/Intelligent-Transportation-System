from env.cityflow_env import CityFlowEnv
from agents.ppo_agent import create_ppo_agent
from training.callbacks import MetricsCallback
from config.load_config import load_config
from pathlib import Path
import random
import numpy as np

cfg = load_config()

SEED = cfg["general"]["seed"]
BASELINE_NAME = cfg["general"]["baseline_name"]

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / BASELINE_NAME
RESULT_PATH.mkdir(parents=True, exist_ok=True)


def train():

    random.seed(SEED)
    np.random.seed(SEED)

    env = CityFlowEnv()
    env.seed(SEED)

    model = create_ppo_agent(env, seed=SEED)

    callback = MetricsCallback(
        save_path=RESULT_PATH,
        check_freq=5000
    )

    model.learn(
        total_timesteps=cfg["training"]["total_timesteps"],
        callback=callback
    )

    model.save(str(RESULT_PATH / "ppo_traffic"))

    print("Training complete.")