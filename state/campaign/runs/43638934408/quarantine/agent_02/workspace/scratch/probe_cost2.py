import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)
w = np.array([s.weight for s in sig])
vals, counts = np.unique(w, return_counts=True)
print("weight distribution:", dict(zip(vals.tolist(), counts.tolist())))
print("first close", closes[0], "last close", closes[-1])
print("first 10 closes", closes[:10])
# trailing 5-day sign sample
for i in [10, 100, 500, 1000, 2000, 3000, 4000]:
    ret = float(np.mean(np.log(closes[i - 4 : i + 1])))
    print(f"  i={i} close={closes[i]:.2f} 5d_logret={ret:+.5f} weight={sig[i].weight}")