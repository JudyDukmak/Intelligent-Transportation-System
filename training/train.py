from env.cityflow_env import CityFlowEnv
from agents.ppo_agent import create_ppo_agent
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "baseline2"
RESULT_PATH.mkdir(parents=True, exist_ok=True)

def train():
    env = CityFlowEnv()
    model = create_ppo_agent(env)
    model.learn(total_timesteps=200000)  # long training
    model.save(str(RESULT_PATH / "ppo_traffic"))
    print("Training complete. Model saved to", RESULT_PATH)