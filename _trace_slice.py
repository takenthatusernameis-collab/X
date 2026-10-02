import warnings, sys
from pathlib import Path
import numpy as np
warnings.filterwarnings('error', category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

np.random.seed(42)
bars = bt.generate_bars(2500, regimes=[
    bt.Regime(drift_annual=0.03, vol_annual=0.18, intraday_range_scale=0.015),
    bt.Regime(drift_annual=-0.01, vol_annual=0.35, intraday_range_scale=0.030),
], p_transition=0.008, start_price=100.0, seed=42)

def ma_signals(closes, fast, slow):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(int(slow) - 1, n):
        fast_ma = np.mean(closes[i - int(fast) + 1 : i + 1])
        slow_ma = np.mean(closes[i - int(slow) + 1 : i + 1])
        out[i] = bt.Signal(date=i + 1, weight=1.0 if fast_ma > slow_ma else -1.0)
    return out

grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)), multipliers=(0.5, 1.0, 2.0))
print("grid:", grid)
for p in grid:
    try:
        s = ma_signals(bars.closes_array(), p["fast"], p["slow"])
        print(p, "ok", len(s))
    except RuntimeWarning as e:
        print(p, "ERROR:", e)
