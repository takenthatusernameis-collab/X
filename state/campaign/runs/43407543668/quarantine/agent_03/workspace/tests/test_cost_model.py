"""Characterization tests for the engine's transaction-cost accounting.

These tests pin down how `BacktestConfig` costs behave over a single completed
round trip (long one bar, flat the next). They are characterization tests: they
describe the current engine, they do not assert that the behavior is desirable.

Motivation: task R-001 (2026-11-13) found that the cost-sensitivity check under
the repository's "realistic cost" model showed ~zero cost drag. The cause is
that additive fill-price slippage is applied identically on entry and exit and
therefore cancels over a completed round trip, while proportional slippage
scales both legs and multiplies P&L by (1+p). Only commission is a genuine
round-trip cost in the current engine. If the engine is later changed to charge
directional slippage, these tests should fail and be updated deliberately.
"""
import unittest

import research.backtest as bt


def _two_bar_run(cfg):
    """Long at bar 0, flat at bar 1; return final equity."""
    bars = [
        bt.Bar(date=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0),
        bt.Bar(date=2, open=110.0, high=110.0, low=110.0, close=110.0, volume=1.0),
    ]
    signals = [bt.Signal(date=1, weight=1.0), bt.Signal(date=2, weight=0.0)]
    res = bt.run_bars(bars, signals, cfg)
    return float(res.equity_curve[-1])


class TestCostAccounting(unittest.TestCase):
    def test_fixed_slippage_cancels_over_round_trip(self):
        base = _two_bar_run(bt.BacktestConfig(initial_capital=1e6))
        slipped = _two_bar_run(bt.BacktestConfig(
            initial_capital=1e6, slippage_cents=2.0))
        self.assertAlmostEqual(base, 1_100_000.0, places=6)
        self.assertAlmostEqual(slipped, base, places=6)

    def test_proportional_slippage_scales_pnl_not_cost(self):
        base = _two_bar_run(bt.BacktestConfig(initial_capital=1e6))
        slipped = _two_bar_run(bt.BacktestConfig(
            initial_capital=1e6, slippage_proportional=0.0005))
        pnl = base - 1e6
        slipped_pnl = slipped - 1e6
        self.assertAlmostEqual(slipped_pnl, pnl * 1.0005, places=6)
        self.assertGreater(slipped_pnl, pnl)

    def test_commission_is_the_only_true_round_trip_cost(self):
        base = _two_bar_run(bt.BacktestConfig(initial_capital=1e6))
        commissioned = _two_bar_run(bt.BacktestConfig(
            initial_capital=1e6, commission_per_trade=2.0))
        self.assertAlmostEqual(base - commissioned, 4.0, places=6)


if __name__ == "__main__":
    unittest.main()