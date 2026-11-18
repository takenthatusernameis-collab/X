"""Diagnostic: is bt.momentum_signals a genuine momentum signal?

Reports, per ticker and lookback, whether the signal ever flips sign and what
quantity it actually thresholds. The framework computes
    ret = mean(log(closes[i-lookback+1 : i+1]))
i.e. the MEAN OF LOG PRICES over the window, then goes long if ret > 0.
For any price series whose adjusted close is > 1.0, mean(log price) > 0 for
every window, so ret > 0 always and the signal is a CONSTANT long (+1).

This diagnostic (a) confirms the degeneracy across the universe, and (b)
contrasts it with the conventional momentum definition
    sign(log(close[i] / close[i-lookback]))
to quantify the difference. This is diagnosis of a candidate-evidence defect,
not a signal-rule tuning step.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

LOOKBACKS = (3, 5, 10)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60


def conventional_momentum_signals(closes, lookback):
    """Standard momentum: sign of the trailing lookback-day log return."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        r = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if r > 0 else -1.0)
    return out


def summarize(sig):
    w = np.array([s.weight for s in sig])
    nz = w[w != 0]
    flips = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
    return sorted(set(float(x) for x in nz)), flips


manifest = load_manifest()
tickers = [e["ticker"] for e in manifest["entries"]]

print("Framework momentum_signals degeneracy check (mean-of-log-price rule):")
print("{:6s} {:>8s} {:>8s} {:>14s} {:>10s} {:>10s}".format(
    "ticker", "pmin", "pmax", "weights(set)", "flips", "minmeanlog"))
for t in tickers:
    bars, _ = bt.load_ticker(t)
    closes = bars.closes_array()
    for lb in LOOKBACKS:
        sig = bt.momentum_signals(closes, lookback=lb)
        wset, flips = summarize(sig)
        # min over windows of mean(log window)
        ml = min(float(np.mean(np.log(closes[i - lb + 1:i + 1])))
                 for i in range(lb, len(closes)))
        print("{:6s} lb{:<2d} {:>8.2f} {:>8.2f} {:>14s} {:>10d} {:>10.3f}".format(
            t, lb, float(closes.min()), float(closes.max()), str(wset), flips, ml))

print("\nConventional momentum (sign of trailing log return) for contrast:")
print("{:6s} {:>10s} {:>10s} {:>10s}".format("ticker", "lb", "flips", "weightset"))
for t in tickers:
    bars, _ = bt.load_ticker(t)
    closes = bars.closes_array()
    for lb in LOOKBACKS:
        sig = conventional_momentum_signals(closes, lb)
        wset, flips = summarize(sig)
        print("{:6s} {:>10d} {:>10d} {:>10s}".format(t, lb, flips, str(wset)))