import sys
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt

for ticker in ["AMZN", "AAPL", "JPM"]:
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    w = np.array([s.weight for s in sig])
    print(ticker, "price range", round(float(closes.min()), 3), round(float(closes.max()), 3))
    print("   weights: n+1=", int(np.sum(w == 1.0)), "n-1=", int(np.sum(w == -1.0)),
          "n0=", int(np.sum(w == 0.0)), "of", len(w))
    # replicate the implementation's ret for a few bars
    rets = [float(np.mean(np.log(closes[i - 5 + 1:i + 1]))) for i in range(5, 40)]
    print("   impl ret[:10]:", [round(r, 4) for r in rets[:10]])
    # correct lookback return
    corr = [float(np.log(closes[i] / closes[i - 5])) for i in range(5, 40)]
    print("   correct ret[:10]:", [round(r, 4) for r in corr[:10]])
    # sign of correct ret distribution over whole series
    sign_corr = np.sign([float(np.log(closes[i] / closes[i - 5])) for i in range(5, len(closes))])
    print("   correct-ret sign: n+=", int(np.sum(sign_corr > 0)), "n-=", int(np.sum(sign_corr < 0)))
    flips = int(np.sum(np.diff(np.sign(w)) != 0))
    print("   weight sign flips:", flips)