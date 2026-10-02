import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

regimes = [
    bt.Regime(drift_annual=0.03, vol_annual=0.18, intraday_range_scale=0.015),
    bt.Regime(drift_annual=-0.01, vol_annual=0.35, intraday_range_scale=0.030),
]

closes_list = []
for r in range(3):
    bars = bt.generate_bars(2500, regimes=regimes, p_transition=0.008, start_price=100.0, seed=42)
    closes_list.append(bars.closes.copy())

print("all equal:", all(np.array_equal(closes_list[0], c) for c in closes_list[1:]))
for i, c in enumerate(closes_list):
    print("run", i, c[:3])
