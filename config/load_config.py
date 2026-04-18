import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "config" / "hyperparams.json"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)