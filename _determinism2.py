"""Check whether generate_bars itself is deterministic."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

def gen():
    bars = bt.generate_bars(2500, regimes=[
        bt.Regime(drift_annual=0.03, vol_annual=0.18, intraday_range_scale=0.015),
        bt.Regime(drift_annual=-0.01, vol_annual=0.35, intraday_range_scale=0.030),
    ], p_transition=0.008, start_price=100.0)
    return bars.closes.copy()

closes1 = gen()
closes2 = gen()
print("closes equal:", np.array_equal(closes1, closes2))
print("first 5 closes run1:", closes1[:5])
print("first 5 closes run2:", closes2[:5])
print("max diff:", np.abs(closes1 - closes2).max())
