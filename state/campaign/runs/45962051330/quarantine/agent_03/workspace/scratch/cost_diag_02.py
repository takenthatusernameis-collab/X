import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import research.backtest as bt

ZERO = bt.BacktestConfig(initial_capital=1e6)
REAL = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005,
)

for ticker in ["AAPL", "AMZN"]:
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    for label, cfg in (("zero", ZERO), ("real", REAL)):
        res = bt.run_bars(list(bars), sig, cfg)
        n_trades = len(res.trades)
        total_comm = sum(f.commission for f in res.trades)
        # slippage is embedded in fill price; estimate slippage notional:
        # recompute gross traded notional
        gross = sum(abs(f.shares) * f.price for f in res.trades)
        eq0 = res.equity_curve[0]
        eqT = res.equity_curve[-1]
        print(f"{ticker} {label}: n_trades={n_trades} total_commission={total_comm:.1f} "
              f"gross_traded_notional={gross:.3e} total_return={(eqT-eq0)/eq0:+.4f}")
    # signal flip count
    flips = sum(1 for i in range(1, len(sig)) if sig[i].weight != sig[i-1].weight)
    print(f"  {ticker}: bars={len(sig)} sign_flips={flips} flip_rate={flips/len(sig):.3f}")