"""Scratch diagnostic: measure turnover / traded notional / commission.

Temporary debug helper for the R-001 cost-sensitivity check. Not part of the
evidence base; safe to delete.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

for ticker in ["AAPL", "AMZN"]:
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)

    for name, cfg in {
        "zero": bt.BacktestConfig(initial_capital=1e6),
        "realistic": bt.BacktestConfig(initial_capital=1e6,
            commission_per_trade=2.0, commission_per_share=0.003,
            slippage_cents=2.0, slippage_proportional=0.0005),
    }.items():
        res = bt.run_bars(list(bars), sig, cfg)
        fills = res.trades
        traded_notional = sum(abs(f.shares) * f.price for f in fills)
        commission = sum(f.commission for f in fills)
        # signed turnover: sum of |delta| in shares
        shares_traded = sum(abs(f.shares) for f in fills)
        n = len(bars)
        print(f"{ticker} {name:10s}: fills={len(fills)} "
              f"shares_traded={shares_traded:.0f} "
              f"traded_notional={traded_notional:.3e} "
              f"commission={commission:.2f} "
              f"final_equity={res.equity_curve[-1]:.0f} "
              f"n_bars={n} fills_per_bar={len(fills)/n:.3f}")

    # Per-fold fill count under realistic cfg
    cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=60,
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005)
    wf = bt.walk_forward(list(bars), sig, train_window=252, test_window=84,
                         warmup=60, overlap_window=60, cfg=cfg)
    fills_per_fold = [f.metrics["n_trades"] for f in wf.folds]
    print(f"  {ticker} realistic walk_forward: n_folds={len(wf.folds)} "
          f"mean n_trades/fold={np.mean(fills_per_fold):.2f} "
          f"first5 n_trades={fills_per_fold[:5]}")
    print()