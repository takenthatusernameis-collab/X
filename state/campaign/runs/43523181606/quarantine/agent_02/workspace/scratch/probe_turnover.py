import sys
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt
from research.data.preflight import load_manifest

manifest = load_manifest()
print("dataset", manifest["dataset_id"], "tickers", len(manifest["entries"]))
for lb in (3, 5, 10):
    tot_turn = []
    flips = []
    for entry in manifest["entries"]:
        bars, _ = bt.load_ticker(entry["ticker"])
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=lb)
        w = np.array([s.weight for s in sig])
        res = bt.run_bars(list(bars), sig, bt.BacktestConfig(initial_capital=1e6))
        gross = sum(abs(f.shares) * f.price for f in res.trades) / 1e6
        tot_turn.append(gross / len(bars))
        flips.append(float((np.diff(w) != 0).sum()) / len(w))
    print("lookback", lb, "turnover/bar median", round(float(np.median(tot_turn)), 4),
          "max", round(max(tot_turn), 4),
          "flip-rate median", round(float(np.median(flips)), 4))

bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
for lb in (3, 5, 10):
    w = np.array([s.weight for s in bt.momentum_signals(closes, lookback=lb)])
    print("AMZN lookback", lb, "long", int((w > 0).sum()), "short", int((w < 0).sum()),
          "changes", int((np.diff(w) != 0).sum()))