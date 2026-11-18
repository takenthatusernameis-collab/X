"""Scratch diagnostic: quantify the actual cost drag per momentum flip.

The engine sizes constant-dollar at initial_capital=1e6, so a fixed $/trade
commission is tiny relative to notional. This script measures:
  (a) the per-flip cost as a fraction of notional for each grid point;
  (b) the total return drag of the harsh fixed-cost level vs zero cost;
  (c) a proportional-slippage stress (5 / 10 / 20 bps) that is the more
      decision-relevant cost model for a daily-rebalanced strategy.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

CAPITAL = 1e6

# (a) per-flip cost as fraction of notional
print("=== (a) per-flip cost as fraction of $1M notional ===")
for label, comm, slip_cents in [("zero", 0, 0), ("retail", 1, 1), ("tight", 5, 2), ("harsh", 10, 5)]:
    # One flip = sell old + buy new = 2 trades; slippage applies to shares at price ~ P.
    # For a $1M notional at price P, shares = 1e6/P. Slippage cost per trade = shares*slip_cents/100.
    # Fixed commission per trade = comm.
    # Round-trip (2 trades) cost = 2*(comm + shares*slip/100) = 2*comm + 2*(1e6/P)*slip/100.
    # As fraction of notional (independent of P): 2*comm/1e6 + 2*slip/100/P... depends on P.
    # Use a representative price P=200 (large-cap): shares=5000.
    P = 200.0
    shares = CAPITAL / P
    rt_fixed = 2 * comm  # dollars
    rt_slip = 2 * shares * slip_cents / 100.0  # dollars
    rt_total = rt_fixed + rt_slip
    frac_notional = rt_total / CAPITAL
    print(f"  {label:8s}: comm=${comm}/trade slip={slip_cents}c/sh | at P={P}: "
          f"round-trip cost ${rt_total:.2f} = {frac_notional*1e4:.2f} bps of notional")

# (b)/(c) empirical drag on AMZN
bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

print("\n=== (b) full-sample total return drag (AMZN lookback 5) ===")
configs = [
    ("zero", bt.BacktestConfig(initial_capital=CAPITAL, commission_per_trade=0, slippage_cents=0)),
    ("harsh_fixed", bt.BacktestConfig(initial_capital=CAPITAL, commission_per_trade=10, slippage_cents=5)),
    ("prop_5bps", bt.BacktestConfig(initial_capital=CAPITAL, slippage_proportional=0.0005)),
    ("prop_10bps", bt.BacktestConfig(initial_capital=CAPITAL, slippage_proportional=0.0010)),
    ("prop_20bps", bt.BacktestConfig(initial_capital=CAPITAL, slippage_proportional=0.0020)),
]
base_ret = None
for label, cfg in configs:
    res = bt.run_bars(list(bars), sig, cfg)
    total = (res.equity_curve[-1] - CAPITAL) / CAPITAL
    if base_ret is None:
        base_ret = total
    print(f"  {label:12s}: total_return {total:+.4f}  drag vs zero {total - base_ret:+.4f}  "
          f"n_fills {len(res.trades)}")

# How many fills per year does the strategy generate? (turnover)
res0 = bt.run_bars(list(bars), sig, configs[0][1])
n_years = len(closes) / 252.0
print(f"\n  bars={len(closes)} (~{n_years:.1f}y), fills={len(res0.trades)} "
      f"-> {len(res0.trades)/n_years:.0f} fills/year")
print("DIAG OK")