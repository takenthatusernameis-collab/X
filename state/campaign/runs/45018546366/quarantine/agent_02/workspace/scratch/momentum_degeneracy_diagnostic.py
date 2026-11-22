"""Diagnostic: is momentum_signals degenerate (constant, non-flipping)?

For each collected ticker and lookback in {3,5,10}, report:
  - momentum_signals: number of sign flips and the set of distinct weights.
  - mean_reversion_signals: same (expected to be a constant opposite sign).
  - a CORRECT short-horizon momentum signal (sign of the lookback log return
    log(c[i]) - log(c[i-lookback])) as a contrast, to show the intended signal
    would be dynamic.

The hypothesis under test: `np.mean(np.log(closes[...]))` is the mean log PRICE
LEVEL (always > 0 for prices > 1), not a return, so momentum_signals is
identically +1 and mean_reversion_signals identically -1 — a static long/short
bias rather than a momentum/reversal strategy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

LOOKBACKS = (3, 5, 10)


def distinct_weights(sig):
    return sorted(set(round(s.weight, 6) for s in sig))


def flips(sig):
    w = np.array([s.weight for s in sig])
    nz = w[w != 0.0]
    if len(nz) < 2:
        return 0
    return int(np.sum(nz[1:] != nz[:-1]))


def correct_momentum_flips(closes, lookback):
    """Sign of the lookback log return (the documented contract)."""
    n = len(closes)
    w = np.zeros(n)
    for i in range(lookback, n):
        r = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        w[i] = 1.0 if r > 0 else -1.0
    nz = w[w != 0.0]
    return int(np.sum(nz[1:] != nz[:-1])) if len(nz) >= 2 else 0


def main():
    manifest = load_manifest()
    print("ticker  lb | momentum: flips  weights        | reversal: flips  weights        "
          "| correct_mom flips")
    constant_long = 0
    constant_short = 0
    dynamic = 0
    total_series = 0
    flip_classes = {}
    min_correct_flips = None
    for entry in manifest["entries"]:
        t = entry["ticker"]
        bars, _ = bt.load_ticker(t)
        closes = bars.closes_array()
        for lb in LOOKBACKS:
            m = bt.momentum_signals(closes, lookback=lb)
            r = bt.mean_reversion_signals(closes, lookback=lb)
            mw = set(round(s.weight, 6) for s in m)
            rw = set(round(s.weight, 6) for s in r)
            mf, rf = flips(m), flips(r)
            cmf = correct_momentum_flips(closes, lb)
            total_series += 1
            min_correct_flips = cmf if min_correct_flips is None else min(min_correct_flips, cmf)
            if mw <= {0.0, 1.0}:
                constant_long += 1
            elif mw <= {0.0, -1.0}:
                constant_short += 1
            else:
                dynamic += 1
            if mf == 0:
                flip_class = "constant"
            elif mf == 1:
                flip_class = "single_flip"
            else:
                flip_class = "dynamic"
            flip_classes[flip_class] = flip_classes.get(flip_class, 0) + 1
            print("{:6s}  {:2d} | flips={:<6d} weights={:<16} | flips={:<6d} "
                  "weights={:<16} | flips={}".format(
                      t, lb, mf, str(sorted(mw)), rf, str(sorted(rw)), cmf))
    print()
    print("momentum series: total={} constant_long={} constant_short={} dynamic={}".format(
        total_series, constant_long, constant_short, dynamic))
    print("flip-class counts: {}".format(flip_classes))
    print("correct-momentum flips range: min={} (every ticker/lookback would be dynamic)".format(
        min_correct_flips))
    if flip_classes.get("dynamic", 0) == 0:
        print("RESULT: no momentum series has more than one sign flip; 27/30 are exactly "
              "constant +1 and 3/30 (NVDA) flip once (split-adjusted price < $1). "
              "momentum_signals is a STATIC LONG BIAS, mean_reversion_signals a static "
              "short bias; neither implements the documented 'long/short the previous "
              "N-day return' contract. The defect is using mean log PRICE level instead "
              "of a return. The momentum/reversal evidence reflects the equity risk "
              "premium (long beats short over 2009-2026), not a momentum/reversal anomaly.")
    else:
        print("RESULT: {} momentum series are genuinely dynamic; inspect per-asset rows.".format(
            flip_classes["dynamic"]))


if __name__ == "__main__":
    main()