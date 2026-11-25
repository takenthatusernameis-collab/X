"""Unit tests for the momentum cost-sensitivity helpers.

These lock the predeclared cost-grid semantics (zero-cost reference, realistic
1x model, and integer scaling) and the cost-robustness classification rule so
the R-001 artifact cannot silently change meaning.
"""
from __future__ import annotations

import unittest

import research.backtest as bt
from research.checks.momentum_cost_sensitivity import (
    BASE_REALISTIC_COST,
    COST_MULTIPLIERS,
    cell_is_edge,
    classify_cost_robustness,
    cost_config,
)


class TestCostConfig(unittest.TestCase):
    def test_zero_multiplier_is_zero_cost_reference(self):
        cfg = cost_config(0.0)
        self.assertEqual(cfg.commission_per_trade, 0.0)
        self.assertEqual(cfg.commission_per_share, 0.0)
        self.assertEqual(cfg.slippage_cents, 0.0)
        self.assertEqual(cfg.slippage_proportional, 0.0)

    def test_one_multiplier_is_realistic_model(self):
        cfg = cost_config(1.0)
        self.assertAlmostEqual(cfg.commission_per_trade,
                               BASE_REALISTIC_COST["commission_per_trade"])
        self.assertAlmostEqual(cfg.commission_per_share,
                               BASE_REALISTIC_COST["commission_per_share"])
        self.assertAlmostEqual(cfg.slippage_cents,
                               BASE_REALISTIC_COST["slippage_cents"])
        self.assertAlmostEqual(cfg.slippage_proportional,
                               BASE_REALISTIC_COST["slippage_proportional"])

    def test_multiplier_scales_every_component(self):
        for m in COST_MULTIPLIERS:
            cfg = cost_config(m)
            self.assertAlmostEqual(cfg.commission_per_trade,
                                   BASE_REALISTIC_COST["commission_per_trade"] * m)
            self.assertAlmostEqual(cfg.commission_per_share,
                                   BASE_REALISTIC_COST["commission_per_share"] * m)
            self.assertAlmostEqual(cfg.slippage_cents,
                                   BASE_REALISTIC_COST["slippage_cents"] * m)
            self.assertAlmostEqual(cfg.slippage_proportional,
                                   BASE_REALISTIC_COST["slippage_proportional"] * m)

    def test_zero_cost_matches_default_config_costs(self):
        # The 0x reference must be the same cost model as the admitted run.
        default = bt.BacktestConfig()
        cfg0 = cost_config(0.0)
        self.assertEqual(cfg0.commission_per_trade, default.commission_per_trade)
        self.assertEqual(cfg0.commission_per_share, default.commission_per_share)
        self.assertEqual(cfg0.slippage_cents, default.slippage_cents)
        self.assertEqual(cfg0.slippage_proportional, default.slippage_proportional)


class TestCellIsEdge(unittest.TestCase):
    def test_positive_regime_stable_is_edge(self):
        self.assertTrue(cell_is_edge([0.05, 0.06, 0.07], "REGIME_STABLE"))

    def test_regime_stable_with_negative_median_is_not_edge(self):
        self.assertFalse(cell_is_edge([-0.01, 0.06, 0.07], "REGIME_STABLE"))

    def test_other_verdicts_are_not_edge(self):
        self.assertFalse(cell_is_edge([0.05, 0.06], "REGIME_DEPENDENT"))
        self.assertFalse(cell_is_edge([0.05, 0.06], "CONSISTENT_WITH_NOISE"))
        self.assertFalse(cell_is_edge([-0.05, -0.06], "REGIME_STABLE_LOSS"))


class TestClassifyCostRobustness(unittest.TestCase):
    def test_robust_when_edge_at_max_cost(self):
        self.assertEqual(
            classify_cost_robustness({0.0: True, 1.0: True, 2.0: True, 5.0: True}),
            "COST_ROBUST")

    def test_moderate_when_edge_at_2x_not_5x(self):
        self.assertEqual(
            classify_cost_robustness({0.0: True, 1.0: True, 2.0: True, 5.0: False}),
            "COST_MODERATE")

    def test_fragile_when_edge_at_realistic_not_2x(self):
        self.assertEqual(
            classify_cost_robustness({0.0: True, 1.0: True, 2.0: False, 5.0: False}),
            "COST_FRAGILE")

    def test_zero_cost_only(self):
        self.assertEqual(
            classify_cost_robustness({0.0: True, 1.0: False, 2.0: False, 5.0: False}),
            "ZERO_COST_ONLY")

    def test_no_edge(self):
        self.assertEqual(
            classify_cost_robustness({0.0: False, 1.0: False, 2.0: False, 5.0: False}),
            "NO_EDGE")

    def test_non_monotone_keys_on_highest_cost(self):
        # Edge at 0x and 5x but not 1x/2x -> COST_ROBUST (highest level wins).
        self.assertEqual(
            classify_cost_robustness({0.0: True, 1.0: False, 2.0: False, 5.0: True}),
            "COST_ROBUST")


if __name__ == "__main__":
    unittest.main()