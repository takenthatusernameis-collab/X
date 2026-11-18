"""Diagnostic: verify transaction costs actually change walk_forward medians.

Compares AMZN lookback=5 walk_forward median log return across cost multipliers,
calling bt.walk_forward directly (bypassing stress_segments) so any cost effect
must be visible in the per-fold OOS returns.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
BASE = dict(commission_per_trade=2.0, commission_per_share=0.003,
            slippage_cents=2.0, slippage_proportional=0.0005)


def cfg(m):
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=BASE["commission_per_trade"] * m,
        commission_per_share=BASE["commission_per_share"] * m,
        slippage_cents=BASE["slippage_cents"] * m,
        slippage_proportional=BASE["slippage_proportional"] * m,
        warmup_periods=WARM,
    )


bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

# Count signal flips over the full series (proxy for turnover).
weights = np.array([s.weight for s in sig])
nonflat = weights[weights != 0]
flips = int(np.sum(nonflat[1:] != nonflat[:-1]))
print("AMZN lb5: bars={} nonzero={} sign_flips={}".format(
    len(weights), len(nonflat), flips))

for m in (0.0, 1.0, 2.0, 3.0):
    res = bt.walk_forward(list(bars), sig, train_window=TRAIN, test_window=TEST,
                          warmup=WARM, overlap_window=OVERLAP, cfg=cfg(m))
    folds = res.folds
    logrets = np.array([np.log1p(f.metrics["total_return"]) for f in folds])
    total_trades = sum(f.metrics["n_trades"] for f in folds)
    print("  mult={:>3}: n_folds={} median_log={:+.4f} mean_log={:+.4f} "
          "total_trades={}".format(
              m, len(folds), float(np.median(logrets)),
              float(np.mean(logrets)), total_trades))