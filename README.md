# Traffic Signal Control using CityFlow

## Description
Baseline implementation for single intersection traffic signal control using Hangzhou dataset.

## Overview
This project implements adaptive traffic signal control using PPO reinforcement learning on the CityFlow simulator.

Two baselines are developed:

Baseline 1: Basic PPO with raw state and max-pressure reward
Baseline 2: Improved PPO with normalized state, constrained actions, and multi-objective reward

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
    │   └── ppo_agent.py  # PPO agent definition
    ├── environment/
    │   └── cityflow_env.py  # CityFlow Gym environment
    ├── preprocessing/
    │   └── parser.py  # roadnet parser
    ├── training/
    │   └── train.py  # training loop
    ├── evaluation/
    │   └── evaluate.py  # evaluation + metrics 
    ├── results/   # saved models and outputs 
    ├── main.py     # entry point
    └── config/
        └── config.json
    └── data/
        └── roadnet.json
        └── flow.json


## Baseline 1 (Initial Implementation)
🔹 State Representation
1. Current phase
2. Phase duration
3. Queue length per lane
4. Waiting vehicles per lane
5. Outgoing vehicles
🔹 Action Space
Discrete phase selection
🔹 Reward Function
Max Pressure:
reward = - Σ (queue_in - queue_out)

🔹 Limitations
No normalization → unstable training
No phase constraints → unrealistic switching
Weak reward → poor traffic optimization

--------------------------------------------------------------

## Baseline 2 (Improved Version)
🔹 State Representation 
Normalized state:
1. Phase / total phases
2. Phase time / max time
3. Queue / MAX_QUEUE
4. Waiting / MAX_QUEUE
5. Outgoing / MAX_QUEUE
   
🔹 Action Space 
- Discrete phase selection
- Minimum green time constraint (10 steps)

🔹 Reward Function 
Multi-objective: adding penalty
reward = -pressure - 0.1 * queue - 0.05 * waiting

Improvements:
- Stable training
- Realistic traffic behavior
- Better generalization

🔹 Training Setup
Algorithm: PPO (Proximal Policy Optimization)
Steps:30,000 timesteps(1st baseline), 200,000 timesteps (2nd baseline)
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
Metric	  Baseline 1  	Baseline 2
ATT	        425	         102
AWT	         62          	19
AQL	         93	          50
Throughput  	8          	51

## How to Run
Run inside Docker:
Train:
python main.py --mode train
Evaluate:
python main.py --mode eval
