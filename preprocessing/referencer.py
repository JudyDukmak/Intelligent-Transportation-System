"""
Flow -> ``traffic_arrivals.json`` (delegates to ``baseline.traffic_baseline``).

Run from ``code/``::

    python -m baseline.traffic_baseline preprocess
    python preprocessing/Baseline.py
"""
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from baseline.traffic_baseline import main_preprocess

if __name__ == "__main__":
    main_preprocess()
