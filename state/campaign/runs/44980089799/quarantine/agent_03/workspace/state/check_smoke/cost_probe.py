import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

COST = {
    "zero": dict(),
    "base": dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005),
    "stress2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
                     slippage_cents=4.0, slippage_proportional=0.001),
}

for ticker in ["AMZN", "AAPL"]:
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    null = bt.random_signals(len(closes), seed=42)
    print("==", ticker, "==")
    for name, c in COST.items():
        rc = bt.run_bars(list(bars), sig, bt.BacktestConfig(initial_capital=1e6, **c))
        rn = bt.run_bars(list(bars), null, bt.BacktestConfig(initial_capital=1e6, **c))
        tc = (rc.equity_curve[-1] - 1e6) / 1e6
        tn = (rn.equity_curve[-1] - 1e6) / 1e6
        comm = sum(f.commission for f in rc.trades)
        commn = sum(f.commission for f in rn.trades)
        print("  {:9s} cand_ret={:+.4f} trades={:5d} comm=${:,.0f} | null_ret={:+.4f} trades={:5d} comm=${:,.0f}".format(
            name, tc, len(rc.trades), comm, tn, len(rn.trades), commn))