import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
print("close range", round(float(closes.min()), 2), round(float(closes.max()), 2))
sig = bt.momentum_signals(closes, lookback=5)
w = np.array([s.weight for s in sig])
print("weight counts", {k: int(v) for k, v in zip(*np.unique(w, return_counts=True))})
print("sign flips", int(np.sum(np.diff(w) != 0)))

cfg = bt.BacktestConfig()
res = bt.run_bars(list(bars), sig, cfg)
notionals = np.array([abs(f.shares) * closes[f.date - 1] for f in res.trades])
print("n_fills", len(notionals))
print("notional percentiles", [round(float(np.percentile(notionals, p)), 1) for p in (1, 25, 50, 75, 99)])
for f in res.trades[:8]:
    print("fill", f.date, "shares", round(f.shares, 2), "price", round(f.price, 2),
          "close", round(float(closes[f.date - 1]), 2), "notional", round(abs(f.shares) * closes[f.date - 1], 1))