"""Quick verification that the momentum signal now computes real returns."""
import sys
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt
sys.path.insert(0, "/home/runner/work/X/X/research/data")
from preflight import load_manifest

manifest = load_manifest()
bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)
ws = [int(s.weight) for s in sig[5:]]
from collections import Counter
print("sign counts:", Counter(ws))
cur, start = ws[0], 0
runs = []
for i in range(1, len(ws)):
    if ws[i] != cur:
        runs.append((cur, i - start))
        cur, start = ws[i], i
runs.append((cur, len(ws) - start))
print("first 10 sign-runs (weight, length):", runs[:10])

rsv = bt.mean_reversion_signals(closes, lookback=5)
rws = [int(s.weight) for s in rsv[5:]]
rsv_corr = float(np.corrcoef(ws, rws)[0, 1])
print("momentum/reversal weight correlation:", round(rsv_corr, 3))

n = len(ws)
assert Counter(ws).most_common(1)[0][1] < n, "signal still constant!"
assert len(set(ws)) == 2, "signal must flip sign"
print("OK: momentum signal is no longer constant; it flips sign {} times in {} bars.".format(len(runs), n))
