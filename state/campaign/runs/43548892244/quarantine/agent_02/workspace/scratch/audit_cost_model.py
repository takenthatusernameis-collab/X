"""Read-only audit of the repository's realistic cost model.

Purpose: the cost-sensitivity artifact uses the existing cost vector
(commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0,
slippage_proportional=0.0005). This audit records what that vector actually
charges, because the classification depends on it.

Two findings are quantified:

1. Round-trip slippage cancellation. The engine prices every fill at
   ``close + slippage`` (engine.py `_run_bars`), so a buy and the matching sell
   both receive the same price adjustment. For a completed round trip the
   slippage terms cancel exactly, and proportional slippage multiplies the price
   change instead of subtracting a cost. Only commissions are a genuine cost.
2. Gross intended slippage vs realized slippage cash impact on the collected
   universe, so the understatement is measured rather than asserted.

No research file is modified; this is durable scratch evidence for the handoff.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

COST = dict(commission_per_trade=2.0, commission_per_share=0.003,
            slippage_cents=2.0, slippage_proportional=0.0005)


def round_trip_demo() -> None:
    """Buy one share at bar 0 and sell it at bar 1: slippage must cancel."""
    bars = [
        bt.Bar(date=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1e6),
        bt.Bar(date=2, open=110.0, high=110.0, low=110.0, close=110.0, volume=1e6),
    ]
    sig = [bt.Signal(date=1, weight=0.0), bt.Signal(date=2, weight=0.0)]
    sig[0] = bt.Signal(date=1, weight=1.0)
    sig[1] = bt.Signal(date=2, weight=0.0)
    for label, kw in (
        ("no costs", {}),
        ("fixed slippage 2c", dict(slippage_cents=2.0)),
        ("proportional 5bps", dict(slippage_proportional=0.0005)),
        ("full realistic vector", dict(COST)),
    ):
        cfg = bt.BacktestConfig(initial_capital=100_000.0, **kw)
        res = bt.run_bars(bars, sig, cfg)
        pnl = res.equity_curve[-1] - 100_000.0
        print("  {:24s} round-trip PnL={:+.6f} fills={}".format(
            label, pnl, [(f.shares, round(f.price, 6), round(f.commission, 6)) for f in res.trades]))


def full_sample_audit() -> None:
    for ticker in ("AMZN", "JPM"):
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, 5)
        cfg = bt.BacktestConfig(initial_capital=1e6, **COST)
        res = bt.run_bars(list(bars), sig, cfg)
        notional = sum(abs(f.shares) for f in res.trades)
        commission = sum(f.commission for f in res.trades)
        slippage_cash = sum(-f.shares * (f.price - c)
                            for f, c in zip(res.trades, closes[[b.date - 1 for b in res.trades]]))
        gross_slippage = sum(0.0005 * c * abs(f.shares)
                             for f, c in zip(res.trades, closes[[b.date - 1 for b in res.trades]]))
        print("  {}: fills={} traded_notional={:.0f} ({:.1f}x capital)".format(
            ticker, len(res.trades), notional, notional / 1e6))
        print("    commission={:.0f} ({:.4%} of capital) | engine slippage cash impact={:.0f} "
              "({:.4%}) | gross intended slippage charge={:.0f} ({:.4%})".format(
                  commission, commission / 1e6, slippage_cash, slippage_cash / 1e6,
                  gross_slippage, gross_slippage / 1e6))


def main() -> int:
    print("=== Round-trip slippage test (engine.py `_run_bars` fill pricing) ===")
    round_trip_demo()
    print("\n=== Full-sample cost decomposition, lookback 5 ===")
    full_sample_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())