# Traffic Signal Control using CityFlow

## Description
Baseline implementation for single intersection traffic signal control using Hangzhou dataset.

## Overview
This project implements reinforcement learning-based traffic signal control using the CityFlow simulator on a single intersection (Hangzhou dataset).

We developed two baselines:

Baseline 1: Simple RL environment with basic state and reward
Baseline 2 (Improved): PPO-based model with normalized state, pressure-based reward, and full training/evaluation pipeline

The goal is to minimize congestion, waiting time, and travel time while maximizing throughput.

## Project Pipeline
1-Load traffic simulation (CityFlow)
2-Extract state from intersection
3-RL agent selects traffic phase
4-Apply action to simulator
5-Compute reward
6-Train policy (PPO)
7-Evaluate performance metrics

  
## Folder Structure
code/
└── 
    ├── agents/
    │   └── ppo_agent.py
    ├── environment/
    │   └── cityflow_env.py
    ├── preprocessing/
    │   └── parser.py
    ├── training/
    │   └── train.py
    ├── evaluation/
    │   └── evaluate.py
    ├── results/
    │   └── baseline2/
    ├── main.py
    └── config/
        └── config.json
    └── data/
        └── roadnet.json
        └── flow.json


## Baseline 1 (Initial Implementation)
🔹 State Representation
Raw lane vehicle counts
No normalization
🔹 Action Space
Discrete traffic light phases
🔹 Reward Function
Based on queue or waiting vehicles

🔹 Limitations
No normalization → unstable training
No phase constraints → unrealistic switching
Weak reward → poor traffic optimization
No evaluation metrics

## Baseline 2 (Improved Version)
🔹 State Representation (Enhanced)
We redesigned the state to include:
This is the main contribution of the project.
state = [
    normalized_current_phase,
    normalized_phase_time,
    for each incoming lane:
        (queue_length, waiting_count, outgoing_flow)
]
Improvements:
Normalized values → stable learning
Includes traffic dynamics (incoming + outgoing)
Encodes temporal info (phase time)
🔹 Action Space (Same but Controlled)
action ∈ {traffic light phases}
Enhancement:
Added minimum green time constraint (10 steps)

🔹 Reward Function (Major Upgrade)
We implemented a pressure-based reward:
pressure = Σ(incoming vehicles - outgoing vehicles)
reward = -pressure - 0.1 * queue_length - 0.05 * waiting_time
Why:
Encourages traffic flow balancing
Penalizes congestion
Inspired by state-of-the-art methods (PressLight)

🔹 Training Setup
Algorithm: PPO (Proximal Policy Optimization)
Steps: 200,000 timesteps
Action interval: every 10 simulation steps
Episode length: 3600 steps (1 hour simulation)

🔹Evaluation Metrics:
We evaluate using:
Metric	Description
ATT	Average Travel Time
AQL	Average Queue Length
AWT	Average Waiting Time
Throughput	Vehicles completed

## Results
ATT = 102.82
AQL = 50.8
AWT = 19.47
Throughput = 51

## How to Run
Train:
python main.py --mode train
Evaluate:
python main.py --mode eval
