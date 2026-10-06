"""Diagnostic: check whether costs are applied in walk-forward, and inspect the
coin-flip null distribution for one segment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt
from research.data.preflight import load_manifest

MANIFEST = load_manifest()

bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()

labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
sig = bt.momentum_signals(closes, lookback=5)

# Find the first segment (>= 400 bars)
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

for lab, cfg in [("NO_COST", bt.BacktestConfig()),
                 ("BASE", bt.BacktestConfig(slippage_cents=0.5,
                                            slippage_proportional=0.001)),
                 ("SEVERE", bt.BacktestConfig(slippage_cents=2.0,
                                              slippage_proportional=0.005))]:
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=252, test_window=84,
        warmup=60, overlap_window=60, cfg=cfg)
    total_rets = [f.metrics["total_return"] for f in res.folds]
    log_rets = [np.log1p(np.clip(r, -1.0 + 1e-12, None)) for r in total_rets]
    full_res = bt.run_bars(seg_bars, seg_signals, cfg)
    seg_fills = [(f.date, f.shares, f.price, f.commission)
                 for f in full_res.trades if s <= f.date <= e]
    total_commission = sum(x[3] for x in seg_fills)
    print(f"\n{lab}: n_folds={len(res.folds)} median_log={np.median(log_rets):+.5f} "
          f"mean_log={np.mean(log_rets):+.5f} n_trades={len(seg_fills)} "
          f"total_commission={total_commission:.2f} "
          f"first3_log={log_rets[:3]}")
