"""Measure the actual per-fold transaction-cost drag on the momentum candidate.

For AMZN lookback=5, run walk_forward under zero and conservative costs and
print paired per-fold total returns and their differences, plus the mean/median
difference. Reconciles the small per-fold drag with the larger full-sample
compounded drag.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt

WARM = 60
ZERO = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
CONS = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM,
                         commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.0010)

bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

rz = bt.walk_forward(list(bars), sig, 252, 84, warmup=WARM, overlap_window=60, cfg=ZERO)
rc = bt.walk_forward(list(bars), sig, 252, 84, warmup=WARM, overlap_window=60, cfg=CONS)
tz = [f.metrics["total_return"] for f in rz.folds]
tc = [f.metrics["total_return"] for f in rc.folds]
n = min(len(tz), len(tc))
diffs = [tc[i] - tz[i] for i in range(n)]
print("n_folds zero={} cons={}".format(len(tz), len(tc)))
print("first 8 paired (zero, cons, diff):")
for i in range(8):
    print("  {:2d} {:+.6f} {:+.6f} {:+.6f}".format(i, tz[i], tc[i], diffs[i]))
print("mean per-fold drag (cons-zero): {:+.6f}".format(float(np.mean(diffs))))
print("median per-fold drag:          {:+.6f}".format(float(np.median(diffs))))
print("min/max per-fold drag:         {:+.6f} / {:+.6f}".format(min(diffs), max(diffs)))
lz = np.log1p(np.clip(tz, -1 + 1e-12, None))
lc = np.log1p(np.clip(tc, -1 + 1e-12, None))
print("median log ret zero={:.6f} cons={:.6f} diff={:+.6f}".format(
    float(np.median(lz)), float(np.median(lc)), float(np.median(lc) - np.median(lz))))
print("mean   log ret zero={:.6f} cons={:.6f} diff={:+.6f}".format(
    float(np.mean(lz)), float(np.mean(lc)), float(np.mean(lc) - np.mean(lz))))
print("DONE")