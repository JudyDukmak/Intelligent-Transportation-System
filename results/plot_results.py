import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

wait = np.load(ROOT / "results" / "baseline2"/ "wait.npy")
queue = np.load(ROOT / "results" / "baseline2" / "queue.npy")
throughput = np.load(ROOT / "results" / "baseline2" / "throughput.npy")

plt.figure()
plt.plot(wait)
plt.title("Average Waiting Time per Step")
plt.xlabel("Step")
plt.ylabel("Waiting Vehicles")
plt.savefig(ROOT / "results" / "wait_plot.png")

plt.figure()
plt.plot(queue)
plt.title("Queue Length per Step")
plt.xlabel("Step")
plt.ylabel("Vehicles in Queue")
plt.savefig(ROOT / "results" / "queue_plot.png")

plt.figure()
plt.plot(throughput)
plt.title("Throughput Over Time")
plt.xlabel("Step")
plt.ylabel("Total Vehicles Passed")
plt.savefig(ROOT / "results" / "throughput_plot.png")

print("Plots saved in results/")