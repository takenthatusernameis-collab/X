"""Read-only probe: is the coin-flip null monotonic in transaction costs?

For one segment, recompute the same random signal (same seed) at zero, 1x and 5x
and compare fold-level total returns element by element. Costs must never raise
a fold's total return for an identical fill sequence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest.perturbation import random_signals

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEGMENT_BARS = 60, 4, 400
LEVELS = {
    "zero": dict(),
    "1x": dict(commission_per_trade=2.0, commission_per_share=0.003,
               slippage_cents=2.0, slippage_proportional=0.0005),
    "5x": dict(commission_per_trade=10.0, commission_per_share=0.015,
               slippage_cents=10.0, slippage_proportional=0.0025),
}


def main() -> int:
    for ticker, lookback in (("NVDA", 3), ("AMZN", 10)):
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        segs = []
        cur = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != cur:
                if i - start >= MIN_SEGMENT_BARS:
                    segs.append((cur, start, i))
                cur = labels[i]
                start = i
        if len(labels) - start >= MIN_SEGMENT_BARS:
            segs.append((cur, start, len(labels)))
        name, s, e = segs[0]
        seg_bars = list(bars)[s:e]
        n = e - s
        seed = 42 + lookback * 1000
        sig = random_signals(n, seed=seed)
        print("{} lb{} segment {} bars={} seed={} weights={}".format(
            ticker, lookback, name, n, seed,
            {float(w): int(sum(1 for x in sig if x.weight == w)) for w in (-1.0, 0.0, 1.0)}))
        base = None
        for lvl, kw in LEVELS.items():
            cfg = bt.BacktestConfig(initial_capital=1e6, **kw)
            res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                                  warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
            tr = np.array([f.metrics["total_return"] for f in res.folds], dtype=np.float64)
            if base is None:
                base = tr
                print("  {:4s} folds={} first5={}".format(lvl, len(tr), np.round(tr[:5], 5)))
            else:
                delta = tr - base
                print("  {:4s} folds={} first5={} max_delta={:+.6f} n_higher={} n_equal={}".format(
                    lvl, len(tr), np.round(tr[:5], 5), float(np.max(delta)),
                    int(np.sum(delta > 1e-12)), int(np.sum(np.abs(delta) <= 1e-12))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())