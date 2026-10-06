"""Diagnostic 3: compare per-fold total_returns for cost vs no-cost, and trace
a trade with correct bar lookup from the full series."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

bars, dates = bt.load_ticker("AAPL")
closes_full = bars.closes_array()
date_to_close = {int(d): float(closes_full[i]) for i, d in enumerate(bars.dates)}

# map date -> close using the BarSequence directly
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
print("Segment:", name, "range", s, e, "n_bars:", e - s)
print("  first bar date", s + 1, "close", closes_full[s],
      "| last bar date", e, "close", closes_full[e - 1])

seg_bars = list(bars)[s:e]
seg_signals = sig[s:e]

def fold_returns(cfg):
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=252, test_window=84,
        warmup=60, overlap_window=60, cfg=cfg)
    total_rets = [float(f.metrics["total_return"]) for f in res.folds]
    log_rets = [np.log1p(np.clip(r, -1.0 + 1e-12, None)) for r in total_rets]
    return total_rets, log_rets

for lab, cfg in [("NO_COST", bt.BacktestConfig()),
                 ("BASE", bt.BacktestConfig(slippage_cents=0.5,
                                            slippage_proportional=0.001)),
                 ("SEVERE", bt.BacktestConfig(slippage_cents=2.0,
                                              slippage_proportional=0.005))]:
    tr, logtr = fold_returns(cfg)
    print(f"\n{lab}: median_log={np.median(logtr):+.6f} "
          f"mean_log={np.mean(logtr):+.6f} "
          f"min_log={np.min(logtr):+.6f} max_log={np.max(logtr):+.6f}")

print("\n=== Per-fold comparison (NO_COST vs SEVERE) ===")
print("  fold  NO_COST      SEVERE       diff(NO-SEV)  cost_penalty")
no_tr, _ = fold_returns(bt.BacktestConfig())
sev_tr, _ = fold_returns(bt.BacktestConfig(
    slippage_cents=2.0, slippage_proportional=0.005))
for i, (a, b) in enumerate(zip(no_tr, sev_tr)):
    penalty = a - b
    print(f"  {i:3d}  {a:+.6f}   {b:+.6f}   {penalty:+.6f}  {penalty:+.6f}")
print(f"\n  All penalties >= 0 (costs must never increase return): "
      f"{all(x >= 0 for x in [a - b for a, b in zip(no_tr, sev_tr)])}")
print(f"  Median penalty: {np.median([a-b for a,b in zip(no_tr,sev_tr) ]):+.6f}")

print("\n=== Trade trace (first trades in segment, correct bar lookup) ===")
full_res = bt.run_bars(seg_bars, seg_signals,
                       bt.BacktestConfig(slippage_cents=2.0,
                                         slippage_proportional=0.005))
for i, f in enumerate(full_res.trades[:12]):
    close = date_to_close.get(f.date)
    if close is None:
        # bar outside original dates? use segment position
        pos = f.date - s - 1
        if 0 <= pos < len(seg_bars):
            close = seg_bars[pos].close
        else:
            close = float("nan")
    expected_fill = close + 2.0 / 100 + close * 0.005
    print(f"  trade {i}: date={f.date} shares={f.shares:+9.1f} "
          f"fill={f.price:.4f} close={close:.4f} exp_fill={expected_fill:.4f} "
          f"delta_slippage={f.price - expected_fill:+.6f}")
