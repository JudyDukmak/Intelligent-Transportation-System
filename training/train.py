from env.cityflow_env import CityFlowEnv
from agents.ppo_agent import create_ppo_agent
from pathlib import Path
import random
import numpy as np

SEED = 42  # you can change it, but keep it fixed
ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "baseline2"
RESULT_PATH.mkdir(parents=True, exist_ok=True)

def train():
    random.seed(SEED)
    np.random.seed(SEED)
    env = CityFlowEnv()
    env.seed(SEED)

    model = create_ppo_agent(env, seed=SEED)

    model.learn(total_timesteps=200000)

    model.save(str(RESULT_PATH / "ppo_traffic"))
    print("Training complete. Model saved to", RESULT_PATH)