"""Scratch diagnostic: why does proportional slippage INCREASE AMZN return?

The diag_cost_drag.py result showed total_return increasing with slippage
(zero +5.5373, prop_20bps +5.5463). That is counterintuitive: slippage
should cost money, not add it. Investigate the fill mechanics.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

CAPITAL = 1e6

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

# Run zero-cost and 20bps, compare fill prices and per-trade P&L
for label, cfg in [
    ("zero", bt.BacktestConfig(initial_capital=CAPITAL, commission_per_trade=0, slippage_cents=0, slippage_proportional=0.0)),
    ("prop_20bps", bt.BacktestConfig(initial_capital=CAPITAL, slippage_proportional=0.0020)),
]:
    res = bt.run_bars(list(bars), sig, cfg)
    total = (res.equity_curve[-1] - CAPITAL) / CAPITAL
    # Look at the first 5 fills: fill price vs close on that date
    print(f"\n=== {label}: total_return {total:+.4f}, n_fills {len(res.trades)} ===")
    close_by_date = {b.date: b.close for b in bars}
    for f in res.trades[:6]:
        c = close_by_date[f.date]
        print(f"  date {f.date}: close {c:.4f}  fill_price {f.price:.4f}  "
              f"slip {f.price - c:+.4f}  shares {f.shares:+.0f}  comm {f.commission:.4f}")

# The key question: when the signal goes LONG, fill price = close + slip (pay more).
# When it goes SHORT, fill price = close + slip (receive less). So slippage always
# hurts. But if the signal is a momentum signal that flips long<->short, and the
# price tends to continue in the signal direction, then a HIGHER fill price on the
# long leg means we buy at a higher price and the subsequent move is measured from
# that higher base... but that should still hurt.

# Let's check: does the equity curve use close (not fill price) for marking?
# equity = cash + shares * close. Fill at close+slip means cash is reduced by
# shares*(close+slip). So we start the position with less cash and same shares.
# The mark-to-market gain/loss from close_t to close_{t+1} is shares*(close_{t+1}-close_t),
# independent of fill price. But the CASH outlay differs. So over many flips,
# the cumulative slippage is sum(shares*slip). That is always a COST.

# The ONLY way slippage increases return is if the share counts differ: a higher
# fill price -> fewer shares bought for the same notional -> smaller position ->
# smaller gains AND smaller losses. If the strategy has net gains, smaller position
# = smaller gains = LOWER return. So that doesn't explain it either.

# Check: is target_shares computed from initial_capital (constant) so it's the
# SAME share count regardless of fill price?
# target_shares = weight * target_exposure * initial_capital / bar.close
# Yes! shares depend on bar.close, NOT on fill_price. So share count is identical.
# The only difference is cash outlay. So slippage MUST reduce return.
# Unless... the equity curve marking uses fill price somewhere. Let's check the
# engine: equity_curve[i] = cash + shares * bar.close. cash was reduced by
# delta*fill_price. So higher fill_price -> lower cash -> lower equity. Always a cost.

# So why did prop_20bps show +0.0091 higher return? Let's recompute carefully and
# check for a bug in my diagnostic (maybe slippage_proportional is not being passed
# through walk_forward / run_bars correctly, or the comparison is wrong).

print("\n=== Re-check: run_bars directly with explicit configs ===")
for label, cfg in [
    ("zero", bt.BacktestConfig(initial_capital=CAPITAL, slippage_proportional=0.0)),
    ("prop_20bps", bt.BacktestConfig(initial_capital=CAPITAL, slippage_proportional=0.0020)),
]:
    res = bt.run_bars(list(bars), sig, cfg)
    total = (res.equity_curve[-1] - CAPITAL) / CAPITAL
    print(f"  {label}: equity[-1]={res.equity_curve[-1]:.4f} total_return={total:+.6f}")

# Check if slippage_proportional is actually in BacktestConfig
import dataclasses
print("\nBacktestConfig fields:", [f.name for f in dataclasses.fields(bt.BacktestConfig)])
print("slippage_proportional default:", bt.BacktestConfig().slippage_proportional)
print("DIAG2 OK")