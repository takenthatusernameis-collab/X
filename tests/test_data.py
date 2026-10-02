"""Unit tests for synthetic data generation.

Deterministic and runnable with `python -m unittest`.
"""
import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


class TestSyntheticData(unittest.TestCase):
    """Generated bars must be valid, deterministic and regime-diverse."""

    def test_deterministic_seed(self):
        g1 = bt.generate_bars(500, regimes=None, seed=42)
        g2 = bt.generate_bars(500, regimes=None, seed=42)
        self.assertTrue(np.array_equal(g1.closes, g2.closes))
        self.assertTrue(np.array_equal(g1.opens, g2.opens))

    def test_different_seeds_differ(self):
        g1 = bt.generate_bars(500, seed=42)
        g2 = bt.generate_bars(500, seed=43)
        self.assertFalse(np.array_equal(g1.closes, g2.closes))

    def test_no_nan_or_negatives(self):
        g = bt.generate_bars(1000, seed=99)
        for arr, name in [
            (g.closes, "close"),
            (g.highs, "high"),
            (g.lows, "low"),
            (g.opens, "open"),
            (g.volumes, "volume"),
        ]:
            self.assertFalse(np.isnan(arr).any(), name)
            self.assertFalse((arr < 0).any(), name)

    def test_range_consistency(self):
        g = bt.generate_bars(300, seed=13)
        self.assertTrue((g.highs >= g.opens).all())
        self.assertTrue((g.highs >= g.closes).all())
        self.assertTrue((g.lows <= g.opens).all())
        self.assertTrue((g.lows <= g.closes).all())

    def test_positive_prices(self):
        g = bt.generate_bars(300, seed=13)
        self.assertTrue((g.closes > 0).all())

    def test_regime_switching(self):
        regimes = [
            bt.Regime(vol_annual=0.10, drift_annual=0.05),
            bt.Regime(vol_annual=0.40, drift_annual=-0.02),
        ]
        g = bt.generate_bars(2000, regimes=regimes, p_transition=0.02, seed=7)
        # high-vol regime must actually occur
        daily = np.diff(np.log(g.closes)) * np.sqrt(252)
        self.assertGreater(np.std(daily), 0.10)  # at least regime-1 vol present

    def test_ou_mean_reversion(self):
        regimes = [
            bt.Regime(vol_annual=0.15, mean_reversion_speed=0.05, mean_reversion_level=100.0),
        ]
        g = bt.generate_bars(5000, regimes=regimes, p_transition=0.0, seed=1)
        tail = g.closes[-500:]
        # mean-reverting series should hover near 100 over time
        self.assertAlmostEqual(np.mean(tail), 100.0, delta=15)


class TestBarSequence(unittest.TestCase):
    def test_iteration(self):
        g = bt.generate_bars(50, seed=2)
        bars = list(g)
        self.assertEqual(len(bars), 50)
        self.assertIsInstance(bars[0], bt.Bar)

    def test_getitem(self):
        g = bt.generate_bars(50, seed=2)
        b = g[10]
        self.assertEqual(b.close, g.closes[10])
        self.assertEqual(b.date, 11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
