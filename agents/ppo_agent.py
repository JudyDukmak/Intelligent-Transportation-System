from stable_baselines3 import PPO
from config.load_config import load_config

cfg = load_config()


def create_ppo_agent(env, seed=42):

    ppo = cfg["ppo"]

    model = PPO(
        "MlpPolicy",
        env,
        verbose=ppo["verbose"],
        learning_rate=ppo["learning_rate"],
        n_steps=ppo["n_steps"],
        batch_size=ppo["batch_size"],
        gamma=ppo["gamma"],
        gae_lambda=ppo["gae_lambda"],
        clip_range=ppo["clip_range"],
        ent_coef=ppo["ent_coef"],
        vf_coef=ppo["vf_coef"],
        seed=seed
    )

    return model