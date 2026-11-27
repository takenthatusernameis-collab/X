"""Unit tests for the momentum cost-sensitivity helpers.

Deterministic and runnable with `python -m unittest` or pytest. These lock the
pure classification/cost-calibration contract used by
`research/checks/momentum_cost_sensitivity.py`; they do not re-run the real-data
backtest.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.checks import momentum_cost_sensitivity as mcs


class TestReferencePrice(unittest.TestCase):
    def test_reference_price_is_median(self):
        closes = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertEqual(mcs.reference_price(closes), 3.0)


class TestCostConfig(unittest.TestCase):
    def test_cost_config_scales_with_bps(self):
        cfg0 = mcs.cost_config(0, 50.0)
        cfg10 = mcs.cost_config(10, 50.0)
        cfg25 = mcs.cost_config(25, 50.0)
        self.assertEqual(cfg0.commission_per_share, 0.0)
        self.assertAlmostEqual(cfg10.commission_per_share, 0.001 * 50.0)
        self.assertAlmostEqual(cfg25.commission_per_share, 0.0025 * 50.0)
        # Only the accumulating channel is used; slippage stays zero.
        self.assertEqual(cfg10.slippage_proportional, 0.0)
        self.assertEqual(cfg10.commission_per_trade, 0.0)


class TestEdgeStatus(unittest.TestCase):
    def test_positive_regime_stable_is_edge(self):
        sr = {"verdict": "REGIME_STABLE", "medians": [0.05, 0.06]}
        self.assertEqual(mcs.edge_status(sr), "EDGE")

    def test_uniform_negative_is_loss(self):
        sr = {"verdict": "REGIME_STABLE_LOSS", "medians": [-0.05, -0.06]}
        self.assertEqual(mcs.edge_status(sr), "LOSS")

    def test_mixed_sign_regime_stable_is_not_edge(self):
        # REGIME_STABLE but one negative median -> not a positive edge.
        sr = {"verdict": "REGIME_STABLE", "medians": [0.05, -0.01]}
        self.assertEqual(mcs.edge_status(sr), "NO_EDGE")

    def test_consistent_with_noise_is_no_edge(self):
        sr = {"verdict": "CONSISTENT_WITH_NOISE", "medians": [0.01, 0.02]}
        self.assertEqual(mcs.edge_status(sr), "NO_EDGE")


class TestCostSurvival(unittest.TestCase):
    def test_survival_is_max_bps_with_edge(self):
        by_cost = {
            "0": {"verdict": "REGIME_STABLE", "medians": [0.05]},
            "10": {"verdict": "REGIME_STABLE", "medians": [0.04]},
            "25": {"verdict": "REGIME_STABLE", "medians": [0.03]},
            "50": {"verdict": "CONSISTENT_WITH_NOISE", "medians": [0.0]},
            "100": {"verdict": "CONSISTENT_WITH_NOISE", "medians": [0.0]},
        }
        self.assertEqual(mcs.cost_survival_bps(by_cost), 25)

    def test_survival_none_when_no_edge(self):
        by_cost = {str(b): {"verdict": "CONSISTENT_WITH_NOISE", "medians": [0.0]}
                   for b in mcs.COST_BPS}
        self.assertIsNone(mcs.cost_survival_bps(by_cost))


class TestClassify(unittest.TestCase):
    def _profile(self, edge_bps):
        by_lookback = {}
        for lb in mcs.LOOKBACKS:
            by_cost = {}
            for b in mcs.COST_BPS:
                if b in edge_bps:
                    by_cost[str(b)] = {"verdict": "REGIME_STABLE", "medians": [0.05]}
                else:
                    by_cost[str(b)] = {"verdict": "CONSISTENT_WITH_NOISE", "medians": [0.0]}
            by_lookback[str(lb)] = by_cost
        return by_lookback

    def test_cost_robust(self):
        # Edge survives to 100 bps (>= DECISION_BPS).
        self.assertEqual(
            mcs.classify_cost_robustness(self._profile({0, 10, 25, 50, 100})),
            "COST_ROBUST")

    def test_cost_fragile(self):
        # Edge at 0 only, lost before DECISION_BPS.
        self.assertEqual(
            mcs.classify_cost_robustness(self._profile({0})),
            "COST_FRAGILE")

    def test_no_edge(self):
        self.assertEqual(
            mcs.classify_cost_robustness(self._profile(set())),
            "NO_EDGE")


if __name__ == "__main__":
    unittest.main()