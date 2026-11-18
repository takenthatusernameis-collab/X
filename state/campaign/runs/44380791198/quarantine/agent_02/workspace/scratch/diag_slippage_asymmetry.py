"""Scratch diagnostic: confirm the slippage asymmetry hypothesis.

Hypothesis: for a long/short strategy that flips position every bar, slippage
is a COST when opening (buy high) but a BENEFIT when closing a long (sell high)
and when opening a short (sell high = receive more). Over a full round trip the
slippage on the opening and closing legs can cancel or even net positive if the
strategy is structured so that the "close" leg always benefits.

Minimal test: a deterministic 2-bar round trip.
  Bar 0: signal long (weight +1). Buy at close0+slip.
  Bar 1: signal short (weight -1). Sell (close long) at close1+slip.
Cash flow:
  Buy:  -shares*(close0+slip)
  Sell: +shares*(close1+slip)
Net cash change from the round trip: shares*(close1+slip) - shares*(close0+slip)
                                  = shares*(close1 - close0)   [slip cancels!]
Then equity = cash + shares*close1 (now short shares) ...
The slip cancels on the price but the COMMISSION (if any) does not.

So for a pure long->short flip with no commission, slippage is NEUTRAL, not a
cost, because you buy high and sell high by the same amount. This is the root
cause of the anomaly: the engine's symmetric proportional slippage (always added
to the fill price regardless of side) means slippage cancels for a strategy that
always holds a position and flips direction, rather than going flat.

This is a genuine limitation of the cost model for a continuously-invested
long/short strategy. Fixed per-trade commission does NOT cancel (it's always
subtracted), so the harsh_fixed result (-0.0447 drag) is the meaningful cost
signal, while proportional slippage is structurally neutral for this signal.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

# Minimal 2-bar test: long then short, with proportional slippage.
# Construct bars manually: close0=100, close1=110 (up 10%).
bars = [
    bt.Bar(date=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1e6),
    bt.Bar(date=2, open=110.0, high=110.0, low=110.0, close=110.0, volume=1e6),
]
# Signal: long at bar 1, short at bar 2 (flip).
sig = [bt.Signal(date=1, weight=1.0), bt.Signal(date=2, weight=-1.0)]

CAPITAL = 1e6
for label, slip in [("zero", 0.0), ("prop_10bps", 0.0010)]:
    cfg = bt.BacktestConfig(initial_capital=CAPITAL, slippage_proportional=slip, warmup_periods=0)
    res = bt.run_bars(bars, sig, cfg)
    print(f"\n=== 2-bar flip test: {label} (slip={slip}) ===")
    print(f"  equity curve: {res.equity_curve}")
    print(f"  final equity: {res.equity_curve[-1]:.4f}  return: {(res.equity_curve[-1]-CAPITAL)/CAPITAL:+.6f}")
    for f in res.trades:
        print(f"  fill date {f.date}: shares {f.shares:+.0f} price {f.price:.4f} comm {f.commission:.4f}")

# Expected: with slippage, buy at 100*(1+s), sell at 110*(1+s).
# shares = CAPITAL / 100 (from target_shares = weight*capital/bar.close).
# For weight +1 at bar 1: shares = 1e6/100 = 10000. Buy at 100*(1+s).
# cash = 1e6 - 10000*100*(1+s) = 1e6 - 1e6*(1+s) = -1e6*s.
# equity at bar 1 = cash + shares*close1 = -1e6*s + 10000*100 = -1e6*s + 1e6.
# For weight -1 at bar 2: target_shares = -1e6/110 = -9090.9.
# delta = -9090.9 - 10000 = -19090.9 (sell).
# cash += -delta*fill_price = 19090.9 * 110*(1+s) = 19090.9*110 + 19090.9*110*s.
#   = 2.1e6 + 2.1e6*s (approx). Wait, 19090.9*110 = 2.1e6. Yes.
# cash = -1e6*s + 2.1e6 + 2.1e6*s = 2.1e6 + 1.1e6*s.
# shares = -9090.9. equity = cash + shares*close2 = 2.1e6 + 1.1e6*s + (-9090.9)*110
#   = 2.1e6 + 1.1e6*s - 1e6 = 1.1e6 + 1.1e6*s.
# So equity = 1.1e6*(1+s). With slip, equity is HIGHER by 1.1e6*s.
# This confirms: slippage ADDS to final equity for a long->short flip!
print("\n=== Analytical confirmation ===")
print("  For a long->short flip: final equity = 1.1e6 * (1 + slip).")
print("  Slippage INCREASES equity because the sell leg (closing long + opening short)")
print("  receives the higher fill price, and the position is larger on the sell side")
print("  (closing 10000 long + opening 9090 short = 19090 shares sold at the higher price).")
print("  This is a cost-model artifact, not an exploitable edge.")
print("DIAG3 OK")