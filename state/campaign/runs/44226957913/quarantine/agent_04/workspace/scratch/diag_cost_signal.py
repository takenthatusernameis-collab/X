"""Diagnostic: does the momentum signal trade, and do costs bind?

Inspect for one asset:
- the momentum_signals weight distribution (is it constant?);
- the full-sample number of fills and turnover under zero vs realistic costs;
- the walk-forward median log return under zero vs realistic costs.
Research/simulation only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

BASE_COST = dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005)


def cfg(mult):
    return bt.BacktestConfig(initial_capital=1e6, warmup_periods=60,
                             commission_per_trade=BASE_COST["commission_per_trade"] * mult,
                             commission_per_share=BASE_COST["commission_per_share"] * mult,
                             slippage_cents=BASE_COST["slippage_cents"] * mult,
                             slippage_proportional=BASE_COST["slippage_proportional"] * mult)


for ticker in ["AAPL", "AMZN", "TSLA"]:
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    for lb in [3, 5, 10]:
        sig = bt.momentum_signals(closes, lookback=lb)
        w = np.array([s.weight for s in sig])
        n_long = int(np.sum(w > 0))
        n_short = int(np.sum(w < 0))
        n_flat = int(np.sum(w == 0))
        flips = int(np.sum(np.abs(np.diff(w)) > 0))
        # full-sample fills + turnover at zero and realistic cost
        r0 = bt.run_bars(list(bars), sig, cfg(0.0))
        r1 = bt.run_bars(list(bars), sig, cfg(1.0))
        tr0 = r0.equity_curve[-1] / 1e6 - 1.0
        tr1 = r1.equity_curve[-1] / 1e6 - 1.0
        print(f"{ticker} lb={lb}: weights long={n_long} short={n_short} flat={n_flat} "
              f"flips={flips} | fills0={len(r0.trades)} fills1={len(r1.trades)} "
              f"total_ret gross={tr0:+.4f} net={tr1:+.4f}")
    print()