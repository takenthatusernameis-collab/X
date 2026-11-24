import sys
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt

manifest_universe = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","JPM","JNJ","XOM"]
for lb in [3,5,10]:
    print(f"=== lookback {lb} ===")
    for t in manifest_universe:
        bars = bt.load_ticker(t)[0]
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=lb)
        w = np.array([s.weight for s in sig])
        nz = w[w != 0.0]
        pos = float(np.mean(nz > 0)) if len(nz) else float("nan")
        flips = int(np.sum(np.diff(nz) != 0)) if len(nz) > 1 else 0
        # buy-and-hold total return for reference
        bnh = closes[-1]/closes[0] - 1.0
        print(f"  {t:6s} n={len(w)} nonzero={len(nz)} frac_long={pos:.3f} flips={flips} buyhold={bnh:+.3f}")