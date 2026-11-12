"""Verify whether BacktestConfig costs actually reduce an always-long walk-forward.

Runs the same signal (existing momentum_signals == always-long) on AAPL under
zero vs realistic costs and reports fold total-return aggregates and turnover.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

bars = bt.load_ticker("AAPL")[0]
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

ZERO = bt.BacktestConfig(initial_capital=1e6, warmup_periods=60)
REAL = bt.BacktestConfig(initial_capital=1e6, warmup_periods=60,
                         commission_per_trade=2.0, commission_per_share=0.003,
                         slippage_cents=2.0, slippage_proportional=0.0005)
CONS = bt.BacktestConfig(initial_capital=1e6, warmup_periods=60,
                         commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.001)

for name, cfg in (("zero", ZERO), ("realistic", REAL), ("conservative", CONS)):
    wf = bt.walk_forward(list(bars), sig, train_window=252, test_window=84,
                         warmup=60, overlap_window=60, cfg=cfg)
    m = wf.aggregate_metrics
    # full-sample reference for turnover
    fs = bt.run_bars(list(bars), sig, cfg)
    # total cost paid in the full sample
    total_comm = sum(f.commission for f in fs.trades)
    print(f"{name:12s} folds={m['n_folds']:3d} "
          f"median_log={m['median_log_total_return']:+.4f} "
          f"mean_log={m['mean_log_total_return']:+.4f} "
          f"pos={m['fold_count_positive']}/{m['n_folds']} "
          f"full_total_ret={(fs.equity_curve[-1]-1e6)/1e6:+.4f} "
          f"n_trades={len(fs.trades)} total_comm=${total_comm:,.0f}")

# Show a couple of raw fold total returns to confirm per-fold cost impact.
for name, cfg in (("zero", ZERO), ("realistic", REAL)):
    wf = bt.walk_forward(list(bars), sig, train_window=252, test_window=84,
                         warmup=60, overlap_window=60, cfg=cfg)
    print(f"{name}: first 5 fold total_returns = "
          f"{[round(f.metrics['total_return'],4) for f in wf.folds[:5]]}")