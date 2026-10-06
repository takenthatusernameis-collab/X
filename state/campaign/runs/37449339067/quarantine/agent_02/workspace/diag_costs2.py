"""Diagnostic 2: compare per-fold returns and equity curves for cost vs no-cost,
and trace one round trip to see the fill price."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()

labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
sig = bt.momentum_signals(closes, lookback=5)

runs, cur, start = [], labels[0], 0
for i in range(1, len(labels)):
    if labels[i] != cur:
        if i - start >= 400:
            runs.append((cur, start, i))
        cur, start = labels[i], i
if len(labels) - start >= 400:
    runs.append((cur, start, len(labels)))
name, s, e = runs[0]
print("Segment:", name, s, e, "n_bars:", e - s)

seg_bars = list(bars)[s:e]
seg_signals = sig[s:e]

res0 = bt.run_bars(seg_bars, seg_signals, bt.BacktestConfig())
ressev = bt.run_bars(seg_bars, seg_signals,
                     bt.BacktestConfig(slippage_cents=2.0, slippage_proportional=0.005))

print("\nEquity curve check (first 8 bars of segment):")
print("  NO_COST :", [round(float(v), 3) for v in res0.equity_curve[:8]])
print("  SEVERE  :", [round(float(v), 3) for v in ressev.equity_curve[:8]])
print("  diff    :", [round(float(a - b), 3) for a, b in
                zip(res0.equity_curve, ressev.equity_curve)])

print("\nTrades at bars s+5..s+10 (full_res trades):")
for f in ressev.trades:
    if s + 4 <= f.date <= s + 10:
        bar = seg_bars[f.date - 1]
        print(f"  date={f.date} shares={f.shares:+.1f} price={f.price:.4f} "
              f"(close={bar.close:.4f}) commission={f.commission:.4f} "
              f"slippage={(f.price - bar.close):.6f}")

print("\nWalk-forward per-fold total_return:")
print("  label            NO_COST      SEVERE       diff(NO-SEV)")
for i, f in enumerate(res0.folds):
    sev = ressev.folds[i]
    print(f"  fold {i:2d}  {f.metrics['total_return']:+.5f}   "
          f"{sev.metrics['total_return']:+.5f}   "
          f"{f.metrics['total_return'] - sev.metrics['total_return']:+.5f}")

med0 = np.median([f.metrics["total_return"] for f in res0.folds])
medsev = np.median([f.metrics["total_return"] for f in ressev.folds])
print(f"\nMedian total_return: NO_COST={med0:+.5f} SEVERE={medsev:+.5f}")
print("If costs were charged, SEVERE median must be <= NO_COST median.")
