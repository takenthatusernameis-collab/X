import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)
w = [s.weight for s in sig]
print("n bars", len(w))
print("first 40 weights", w[:40])
# count sign flips
flips = sum(1 for i in range(6, len(w)) if w[i] != w[i-1])
print("sign flips after warmup:", flips)

def hold_transform(signals, hold, lookback):
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < lookback:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - lookback) % hold
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out

for hold in (2,3,5):
    eff = hold_transform(sig, hold, 5)
    diff = sum(1 for i in range(len(w)) if eff[i].weight != w[i])
    print("hold", hold, "diff bars vs base:", diff)

# inspect a transition region
for i in range(100, 130):
    print(i, "base", w[i], "h5", hold_transform(sig,5,5)[i].weight)