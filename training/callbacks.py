import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path


class MetricsCallback(BaseCallback):
    def __init__(self, save_path, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = Path(save_path)

        self.timesteps = []
        self.rewards = []

    def _on_step(self):

        if self.n_calls % self.check_freq == 0:

            mean_reward = np.mean(self.locals["rewards"])

            self.timesteps.append(self.num_timesteps)
            self.rewards.append(mean_reward)

            np.save(self.save_path / "train_steps.npy", self.timesteps)
            np.save(self.save_path / "train_rewards.npy", self.rewards)

        return True