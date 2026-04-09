# Traffic Signal Control using CityFlow

## Description
Baseline implementation for single intersection traffic signal control using Hangzhou dataset.

## Structure
- env/ → simulation environment
- evaluation/ → metrics

  
Folder Structure:
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


