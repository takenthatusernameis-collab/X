"""Diagnostic 4: focused check on ONE fold (fold 15) NO vs SEV,
equity-fill consistency, and a full-sample sanity check."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

bars, dates = bt.load_ticker("AAPL")
closes_full = bars.closes_array()
labels = bt.volatility_blocks(closes_full, n_blocks=4, window=60)
sig = bt.momentum_signals(closes_full, lookback=5)

runs, cur, start = [], labels[0], 0
for i in range(1, len(labels)):
    if labels[i] != cur:
        if i - start >= 400:
            runs.append((cur, start, i))
        cur, start = labels[i], i
if len(labels) - start >= 400:
    runs.append((cur, start, len(labels)))
name, s, e = runs[0]
seg_bars = list(bars)[s:e]
seg_signals = sig[s:e]

NO = bt.BacktestConfig()
SEV = bt.BacktestConfig(slippage_cents=2.0, slippage_proportional=0.005)

# Full-sample runs (no warmup): compare equity and fills directly
f0 = bt.run_bars(seg_bars, seg_signals, NO)
fs = bt.run_bars(seg_bars, seg_signals, SEV)

print("Full-sample equity check:")
print(f"  NO  final equity: {f0.equity_curve[-1]:.2f}  n_trades: {len(f0.trades)}")
print(f"  SEV final equity: {fs.equity_curve[-1]:.2f}  n_trades: {len(fs.trades)}")
print(f"  diff (NO-SEV) final: {f0.equity_curve[-1] - fs.equity_curve[-1]:.2f} (must be >= 0)")
total_cost = sum(f.slippage() if hasattr(f, 'slippage') else 0.0 for f in fs.trades)

# slippage paid = fill_price - close summed
nocs = {int(bd): float(c) for bd, c in zip(bars.dates, closes_full)}
sev_cost = 0.0
for f in fs.trades:
    c = nocs.get(f.date)
    if c is None:
        pos = f.date - s - 1
        c = seg_bars[pos].close if 0 <= pos < len(seg_bars) else None
    if c is not None:
        sev_cost += (f.price - c) * abs(f.shares)
print(f"  SEV total slippage dollars: {sev_cost:.2f}")
print(f"  Expected equity gap >= {sev_cost:.2f}; actual gap: {f0.equity_curve[-1] - fs.equity_curve[-1]:.2f}")

# Equity-fill consistency for SEV
bt.check_equity_matches_fills(fs.equity_curve, fs.trades, seg_bars_closes if False else np.array(
    [seg_bars[i].close for i in range(len(seg_bars))], dtype=np.float64), 1e6)
bt.check_equity_matches_fills(f0.equity_curve, f0.trades, np.array(
    [seg_bars[i].close for i in range(len(seg_bars))], dtype=np.float64), 1e6)
print("  equity_matches_fills: PASS for both")

print("\nWalk-forward fold 15 equity curves:")
def wfo(cfg):
    return bt.walk_forward(seg_bars, seg_signals, train_window=252, test_window=84,
                           warmup=60, overlap_window=60, cfg=cfg)
w0, ws = wfo(NO), wfo(SEV)
assert len(w0.folds) == len(ws.folds), (len(w0.folds), len(ws.folds))
f15_0, f15_s = w0.folds[15], ws.folds[15]
print(f"  fold 15: NO total_return={f15_0.metrics['total_return']:+.6f} "
      f"SEV total_return={f15_s.metrics['total_return']:+.6f}")
eq0 = f15_0.equity_curve
eqs = f15_s.equity_curve
diff = eq0 - eqs
print(f"  fold15 eq0 shape: {eq0.shape}, eqs shape: {eqs.shape}")
print(f"  diff range: [{diff.min():+.3f}, {diff.max():+.3f}]")
print(f"  All diffs >= 0: {all(d >= -1e-6 for d in diff)}")
print(f"  fold15 OOS (last 84): NO={eq0[-1]:.2f} vs SEV={eqs[-1]:.2f}, "
      f"gap={eq0[-1]-eqs[-1]:+.2f}")

print("\nTotal fold count:", len(w0.folds))
