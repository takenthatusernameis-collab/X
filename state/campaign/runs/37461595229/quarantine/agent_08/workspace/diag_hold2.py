import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import research.backtest as bt
from research.data.preflight import load_manifest

manifest = load_manifest()
# AMZN sample to trace OOS windows vs signal differences
bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()

def momentum_signals(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out

sig = momentum_signals(closes, 5)
def eff_h(signals, hold):
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < 5:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            out[i] = bt.Signal(date=i + 1, weight=signals[i - (i - 5) % hold].weight)
    return out

for hold in (1, 2, 3, 5):
    e = eff_h(sig, hold)
    diffs = [i for i in range(5, len(e)) if e[i].weight != sig[i].weight]
    print(f"hold={hold}: {len(diffs)} bars where effective != original signal")
    if hold != 1:
        print("  sample diff positions:", diffs[:12])

# OOS test windows for one segment (e.g. AMZN turbulent block 0: bars 0-~1400)
# walk-forward: fold_start advances by 84-60=24; warmup=60, train=252, test=84
# test window = [fold_start+60+252, fold_start+60+252+84)
oos_starts = [fs for fs in range(0, 1400, 24)]
oos_windows = [(fs+312, fs+396) for fs in oos_starts]
def in_any_oos(pos):
    return any(s <= pos < e for s, e in oos_windows)
d2 = [i for i in range(5, len(e)) if eff_h(sig,2)[i].weight != sig[i].weight]
d5 = [i for i in range(5, len(sig)) if eff_h(sig,5)[i].weight != sig[i].weight]
print("AMZN hold=2 diffs in OOS windows:", [p for p in d2 if in_any_oos(p)])
print("AMZN hold=5 diffs in OOS windows:", [p for p in d5 if in_any_oos(p)])
