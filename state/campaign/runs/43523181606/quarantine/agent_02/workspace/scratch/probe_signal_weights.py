import sys
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
for lb in (3, 5, 10, 20, 60):
    sig = bt.momentum_signals(closes, lookback=lb)
    w = np.array([s.weight for s in sig])
    print("lookback", lb, "n", len(w), "n_long", int((w > 0).sum()),
          "n_short", int((w < 0).sum()), "n_flat", int((w == 0).sum()),
          "n_changes", int((np.diff(w) != 0).sum()))

rev = bt.mean_reversion_signals(closes, lookback=5)
w = np.array([s.weight for s in rev])
print("reversal lb5 n_long", int((w > 0).sum()), "n_short", int((w < 0).sum()),
      "n_flat", int((w == 0).sum()))

print("close range", float(closes.min()), float(closes.max()))
print("mean log close", float(np.mean(np.log(closes))))