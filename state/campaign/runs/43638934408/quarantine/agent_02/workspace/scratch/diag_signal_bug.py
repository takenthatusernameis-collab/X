"""Diagnostic: is research.backtest.momentum_signals actually computing the
previous-N-day return, or the mean log PRICE level? (scratch, not a result.)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]

print("ticker  n   +1     -1    0   flips  price_range        code_sign_vs_true_return_mismatches")
for t in UNIVERSE:
    bars, _ = bt.load_ticker(t)
    c = bars.closes_array()
    sig = bt.momentum_signals(c, lookback=5)
    w = np.array([s.weight for s in sig])
    flips = int(np.sum(w[1:] != w[:-1]))
    # independent TRUE 5-day return sign
    true_w = np.zeros(len(c))
    for i in range(5, len(c)):
        true_ret = np.log(c[i]) - np.log(c[i - 5])
        true_w[i] = 1.0 if true_ret > 0 else -1.0
    mismatch = int(np.sum((w[5:] != 0) & (true_w[5:] != 0) & (w[5:] != true_w[5:])))
    print(f"{t:6s} {len(c)} {int((w>0).sum()):5d} {int((w<0).sum()):5d} {int((w==0).sum()):4d} "
          f"{flips:5d}  [{c.min():.2f},{c.max():.2f}]  {mismatch}")

# Direct counterexample on AAPL
bars, _ = bt.load_ticker("AAPL")
c = bars.closes_array()
print("\nAAPL counterexamples (code weight vs true previous-5d return):")
for i in [10, 100, 1000, 2000, 3000, 4000]:
    true_ret = np.log(c[i]) - np.log(c[i - 5])
    code_term = float(np.mean(np.log(c[i - 4 : i + 1])))
    print(f"  i={i:4d} close={c[i]:8.2f} true_5d_ret={true_ret:+.4f} "
          f"code_mean_log_price={code_term:+.4f} code_weight={bt.momentum_signals(c,5)[i].weight:+.0f}")