"""Diagnostic: confirm the realistic cost model materially charges costs and
that the COST_ROBUST result is not a silently-no-op cost config.

Compares full-sample AMZN/JPM momentum (lookback 5) equity and total commission
paid under the zero-cost and realistic-cost BacktestConfig. If the realistic run
charges ~0 commission, the cost-sensitivity result would be an artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

ZERO = bt.BacktestConfig(initial_capital=1e6)
REALISTIC = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)


def main():
    for ticker in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=5)
        for label, cfg in (("zero", ZERO), ("realistic", REALISTIC)):
            res = bt.run_bars(list(bars), sig, cfg)
            total_comm = sum(f.commission for f in res.trades)
            total_ret = (res.equity_curve[-1] - 1e6) / 1e6
            print("  {} {}: fills={} total_commission={:.2f} "
                  "total_return={:+.4f}".format(
                      ticker, label, len(res.trades), total_comm, total_ret))
        # direct delta
        rz = bt.run_bars(list(bars), sig, ZERO)
        rr = bt.run_bars(list(bars), sig, REALISTIC)
        tz = (rz.equity_curve[-1] - 1e6) / 1e6
        tr = (rr.equity_curve[-1] - 1e6) / 1e6
        print("  {} full-sample zero {:.4f} -> realistic {:.4f} (delta {:+.4f})".format(
            ticker, tz, tr, tz - tr))
        print("")


if __name__ == "__main__":
    main()