import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

cfg = bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                        slippage_cents=2.0, slippage_proportional=0.0005)
res = bt.run_bars(list(bars), sig, cfg)

buy_notional = 0.0
sell_notional = 0.0
slippage_cash_impact = 0.0
for i, f in enumerate(res.trades):
    bar_close = closes[f.date - 1]
    notional = abs(f.shares) * bar_close
    if f.shares > 0:
        buy_notional += notional
    else:
        sell_notional += notional
    # cash impact of slippage = -delta*(fill_price - close)  (buy costs, sell gains)
    slippage_cash_impact += -f.shares * (f.price - bar_close)

print("n_fills", len(res.trades))
print("buy_notional", round(buy_notional, 1))
print("sell_notional", round(sell_notional, 1))
print("buy/sell ratio", round(buy_notional / max(sell_notional, 1e-9), 6))
print("slippage_cash_impact (net)", round(slippage_cash_impact, 2))
print("total_commission", round(sum(f.commission for f in res.trades), 2))
print("final_equity", round(float(res.equity_curve[-1]), 2))