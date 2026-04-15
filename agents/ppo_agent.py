from stable_baselines3 import PPO

def create_ppo_agent(env, seed=42):
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        seed=seed   
    )
    return model