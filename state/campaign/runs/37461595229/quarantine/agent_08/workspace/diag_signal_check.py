import sys
sys.path.insert(0, '.')
import numpy as np
import research.backtest as bt

for ticker in ["AAPL", "MSFT", "NVDA", "JPM"]:
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    m = bt.momentum_signals(closes, 5)
    weights = [s.weight for s in m]
    w = np.array(weights[5:])
    print(f"{ticker}: n={len(w)} flips={np.sum(np.diff(w)!=0)} distinct={sorted(set(w))}")
