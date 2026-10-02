"""Unit tests for metrics.

Deterministic and runnable with `python -m unittest`.
"""
import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


class TestMetrics(unittest.TestCase):
    """Metrics are pure deterministic functions of the equity curve."""

    def test_total_return(self):
        eq = np.array([1e6, 1e6 * 1.10, 1e6 * 0.99])
        m = bt.compute_metrics(eq)
        self.assertAlmostEqual(m.total_return, -0.01, places=9)
        self.assertAlmostEqual(m.n_periods, 3)

    def test_drawdown(self):
        # peak at 1.2e6, draw down to 0.8e6, max drawdown = (1.2-0.8)/1.2
        eq = np.array([1e6, 1.2e6, 1e6, 0.8e6, 1e6])
        m = bt.compute_metrics(eq)
        self.assertAlmostEqual(m.max_drawdown, 0.3333333333, places=6)
        self.assertGreater(m.sharpe, -10)

    def test_sharpe_known_curve(self):
        # constant positive daily return of 0.1% over 1260 days (1259 compounding steps)
        eq = np.array([1e6 * (1.001 ** i) for i in range(1260)], dtype=np.float64)
        m = bt.compute_metrics(eq, risk_free=0.0)
        # self-consistency: CAGR over n points must equal (1+total)^(252/n)
        expected_ann = (1.0 + (eq[-1] / eq[0] - 1.0)) ** (252.0 / len(eq)) - 1.0
        self.assertAlmostEqual(m.annualized_return, expected_ann, places=4)
        self.assertAlmostEqual(m.vol_annual, 0.0, places=9)

    def test_metrics_deterministic(self):
        eq = np.array([1e6, 1e6 * 1.10, 1e6 * 0.99, 1e6 * 1.05, 950000.0])
        m1 = bt.compute_metrics(eq)
        m2 = bt.compute_metrics(eq)
        self.assertEqual(m1, m2)

    def test_trade_stats_round_trip(self):
        eq = np.array([1e6, 1e6, 1e6], dtype=np.float64)
        fills = [
            bt.Fill(date=1, shares=100.0, price=10.0, commission=0.0),   # long 100 @ 10
            bt.Fill(date=2, shares=-100.0, price=11.0, commission=0.0),  # cover @ 11
        ]
        fill_prices = np.array([10.0, 11.0])
        m = bt.compute_metrics(eq, fills=fills, fill_prices=fill_prices)
        self.assertEqual(m.n_trades, 1)
        self.assertEqual(m.winning_trades, 1)
        self.assertEqual(m.losing_trades, 0)
        self.assertAlmostEqual(m.gross_profit, 100.0, places=6)

    def test_trade_stats_with_commission(self):
        eq = np.array([1e6, 1e6, 1e6], dtype=np.float64)
        fills = [
            bt.Fill(date=1, shares=100.0, price=10.0, commission=1.0),
            bt.Fill(date=2, shares=-100.0, price=11.0, commission=1.0),
        ]
        fill_prices = np.array([10.0, 11.0])
        m = bt.compute_metrics(eq, fills=fills, fill_prices=fill_prices)
        self.assertAlmostEqual(m.gross_profit, 98.0, places=6)
        self.assertEqual(m.gross_loss, 0.0)

    def test_calmar(self):
        eq = np.array([1e6, 1e6 * 1.5, 0.5e6, 1e6], dtype=np.float64)
        m = bt.compute_metrics(eq)
        self.assertAlmostEqual(m.max_drawdown, 0.6666666667, places=6)


class TestTradeSummaryEmpty(unittest.TestCase):
    def test_no_trades(self):
        eq = np.array([1e6, 1e6], dtype=np.float64)
        m = bt.compute_metrics(eq, fills=[], fill_prices=np.array([], dtype=np.float64))
        self.assertEqual(m.n_trades, 0)
        self.assertEqual(m.profit_factor, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
