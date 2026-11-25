import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

tickers = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","JPM","JNJ","XOM"]
print("ticker  mom_weight_set        rev_weight_set        mom_n_neg  rev_n_pos  min_close")
for t in tickers:
    bars,_ = bt.load_ticker(t)
    c = bars.closes_array()
    mw = set(s.weight for s in bt.momentum_signals(c, lookback=5))
    rw = set(s.weight for s in bt.mean_reversion_signals(c, lookback=5))
    mom_neg = sum(1 for s in bt.momentum_signals(c,5) if s.weight < 0)
    rev_pos = sum(1 for s in bt.mean_reversion_signals(c,5) if s.weight > 0)
    print("{:6s}  {:20s}  {:20s}  {:8d}  {:8d}  {:.4f}".format(
        t, str(mw), str(rw), mom_neg, rev_pos, float(np.min(c))))

# Show the raw quantity the signal thresholds on, for AMZN around a downturn.
bars,_ = bt.load_ticker("AMZN")
c = bars.closes_array()
print("\nAMZN sample mean(log(closes)) over 5-bar windows (should be log-RETURNS):")
for i in (100, 500, 1000, 2000, 3000, 4000):
    raw = float(np.mean(np.log(c[i-4:i+1])))
    correct = float(np.log(c[i]/c[i-5]))
    print("  i={:5d} mean_log_price={:+.4f}  log_return={:+.4f}".format(i, raw, correct))