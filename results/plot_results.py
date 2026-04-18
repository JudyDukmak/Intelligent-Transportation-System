import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config.load_config import load_config

cfg = load_config()
BASELINE_NAME = cfg["general"]["baseline_name"]


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / BASELINE_NAME
PLOT_PATH = RESULT_PATH / "plots"
PLOT_PATH.mkdir(exist_ok=True)

# Evaluation data
wait = np.load(RESULT_PATH / "wait.npy")
queue = np.load(RESULT_PATH / "queue.npy")


# Training data
steps = np.load(RESULT_PATH / "train_steps.npy")
rewards = np.load(RESULT_PATH / "train_rewards.npy")


# -------------------------
# Evaluation plots
# -------------------------

plt.figure(figsize=(10,5))
plt.plot(wait)
plt.title("Waiting Vehicles")
plt.xlabel("Step")
plt.ylabel("Vehicles")
plt.grid()
plt.savefig(PLOT_PATH / "wait_plot.png")


plt.figure(figsize=(10,5))
plt.plot(queue)
plt.title("Queue Length")
plt.xlabel("Step")
plt.ylabel("Vehicles")
plt.grid()
plt.savefig(PLOT_PATH / "queue_plot.png")


# -------------------------
# Training plots
# -------------------------

# Reward vs timesteps
plt.figure(figsize=(10,5))
plt.plot(steps, rewards)
plt.title("Reward vs Timesteps")
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.grid()
plt.savefig(PLOT_PATH / "reward_vs_timesteps.png")


# Convergence
plt.figure(figsize=(10,5))
plt.plot(steps, rewards)
plt.title("Training Convergence")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.grid()
plt.savefig(PLOT_PATH / "convergence.png")


# AWT trend
window = 30
awt_smooth = np.convolve(wait, np.ones(window)/window, mode="valid")

plt.figure(figsize=(10,5))
plt.plot(awt_smooth)
plt.title("AWT Trend")
plt.xlabel("Step")
plt.ylabel("Avg Waiting Time")
plt.grid()
plt.savefig(PLOT_PATH / "awt_trend.png")

print("All plots saved.")