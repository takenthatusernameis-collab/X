import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0,
                   104.0, 103.0, 102.0, 101.0, 100.0])
print("momentum", [s.weight for s in bt.momentum_signals(closes, lookback=3)])
print("reversal", [s.weight for s in bt.mean_reversion_signals(closes, lookback=3)])

bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
for lookback in (3, 5, 10):
    signals = bt.momentum_signals(closes, lookback=lookback)
    print(lookback, {s.weight for s in signals},
          sum(1 for s in signals if s.weight > 0),
          sum(1 for s in signals if s.weight < 0))