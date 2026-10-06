"""Debug: where do costs go in the engine?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import research.backtest as bt

bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

cfg1 = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

res1 = bt.run_bars(list(bars), sig, cfg1)
print("trades:", len(res1.trades))
total_commission = sum(f.commission for f in res1.trades)
total_slippage = sum(f.shares * (2.0/100 + bars[0].close * 0.0005) for f in res1.trades)
print("total commission:", total_commission)
print("total slippage (approx):", total_slippage)
print("equity final zero-based:", res1.equity_curve[-1])
print("equity final vs 1e6 diff:", res1.equity_curve[-1] - 1e6)

# Recompute equity from fills, like check_equity_matches_fills does.
cash = 1e6
shares = 0.0
fills_sorted = sorted(res1.trades, key=lambda f: f.date)
recomputed = np.zeros(len(res1.equity_curve))
fill_index = 0
for i in range(len(res1.equity_curve)):
    bar_date = i + 1
    while fill_index < len(fills_sorted) and fills_sorted[fill_index].date == bar_date:
        f = fills_sorted[fill_index]
        if f.shares > 0:
            cash -= f.shares * f.price + f.commission
        else:
            cash += -f.shares * f.price - f.commission
        shares += f.shares
        fill_index += 1
    recomputed[i] = cash + shares * closes[i]
diff = np.abs(res1.equity_curve - recomputed)
scale = np.maximum(np.abs(res1.equity_curve), np.abs(recomputed))
print("max relative mismatch vs fill-recomputed equity:", (diff / scale).max())
print("recomputed final:", recomputed[-1])
print("cash final:", cash)
print("shares final:", shares)
print("closes[-1]:", closes[-1])
