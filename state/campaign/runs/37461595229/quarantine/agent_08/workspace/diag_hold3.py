import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import research.backtest as bt
from research.data.preflight import load_manifest

manifest = load_manifest()
def momentum_signals(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out

for entry in manifest["entries"]:
    ticker = entry["ticker"]
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = momentum_signals(closes, 5)
    flips_prev = sum(1 for i in range(5, len(sig)) if sig[i].weight != sig[i-1].weight)
    # diffs between effective signal and original for hold 5
    siglist = list(sig)
    eff = list(siglist)
    for i in range(5, len(eff)):
        eff[i] = bt.Signal(date=i+1, weight=siglist[i - (i-5)%5].weight)
    diffs = sum(1 for i in range(5, len(eff)) if eff[i].weight != siglist[i].weight)
    # distinct momentum signs observed
    n_signs = len(set(s.weight for s in sig))
    print(f"{ticker:6s} n_bars={len(sig):5d} flips_vs_prev={flips_prev:3d} eff_h5_diffs={diffs:3d} distinct_signs={n_signs}")
