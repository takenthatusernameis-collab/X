"""Unit tests for the momentum cost-sensitivity check's pure logic.

Covers the cost-config provenance, the signal-validity gate, the positive-edge
rule, and the SURVIVES / ERODED / NO_EDGE classification. Deterministic and
runnable with `python -m unittest`.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys_path_hack = str(_REPO)
import sys
if sys_path_hack not in sys.path:
    sys.path.insert(0, sys_path_hack)

_spec = importlib.util.spec_from_file_location(
    "momentum_cost_sensitivity_under_test",
    _REPO / "research" / "checks" / "momentum_cost_sensitivity.py",
)
mcs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcs)


def _sr(medians, null_medians, verdict):
    return {"medians": list(medians), "null_medians": list(null_medians), "verdict": verdict}


class TestCostConfig(unittest.TestCase):
    def test_realistic_matches_documented_model(self):
        self.assertEqual(mcs.cost_config("realistic"), mcs.REALISTIC_COST)
        self.assertEqual(mcs.REALISTIC_COST["commission_per_trade"], 2.0)
        self.assertEqual(mcs.REALISTIC_COST["commission_per_share"], 0.003)
        self.assertEqual(mcs.REALISTIC_COST["slippage_cents"], 2.0)
        self.assertEqual(mcs.REALISTIC_COST["slippage_proportional"], 0.0005)

    def test_stress2x_is_double_every_parameter(self):
        for k, v in mcs.REALISTIC_COST.items():
            self.assertEqual(mcs.cost_config("stress2x")[k], 2.0 * v)

    def test_zero_has_no_costs(self):
        self.assertEqual(mcs.cost_config("zero"), {})


class TestSignalValidity(unittest.TestCase):
    def _series(self, returns):
        closes = 100.0 * np.exp(np.cumsum(np.asarray(returns, dtype=np.float64)))
        return closes

    def test_true_momentum_signal_is_valid(self):
        # Alternating returns so the true N-day return sign flips.
        rets = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.03] * 20)
        closes = self._series(rets)
        lookback = 3
        sig = [mcs.bt.Signal(date=i + 1, weight=0.0) for i in range(lookback)]
        for i in range(lookback, len(closes)):
            true_ret = np.log(closes[i]) - np.log(closes[i - lookback])
            sig.append(mcs.bt.Signal(date=i + 1, weight=1.0 if true_ret > 0 else -1.0))
        v = mcs.signal_validity(sig, closes, lookback)
        self.assertEqual(v["true_return_sign_agreement"], 1.0)
        self.assertGreater(v["n_flips"], 0)
        self.assertFalse(v["degenerate"])

    def test_constant_long_is_degenerate(self):
        # Rising series with pullbacks; a constant +1 is not momentum.
        rets = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03] * 20)
        closes = self._series(rets)
        lookback = 3
        n = len(closes)
        sig = [mcs.bt.Signal(date=i + 1, weight=1.0) for i in range(n)]
        v = mcs.signal_validity(sig, closes, lookback)
        self.assertEqual(v["dominant_weight_share"], 1.0)
        self.assertLess(v["true_return_sign_agreement"], 0.95)
        self.assertTrue(v["degenerate"])


class TestPositiveEdge(unittest.TestCase):
    def test_positive_stable_edge(self):
        sr = _sr([0.06, 0.07, 0.05], [0.0, 0.01, -0.01], "REGIME_STABLE")
        self.assertTrue(mcs.positive_edge(sr))

    def test_stable_but_gap_nonpositive_is_not_edge(self):
        sr = _sr([0.06, 0.07, 0.05], [0.08, 0.09, 0.07], "REGIME_STABLE")
        self.assertFalse(mcs.positive_edge(sr))

    def test_any_nonpositive_median_is_not_edge(self):
        sr = _sr([0.06, -0.01, 0.05], [0.0, 0.0, 0.0], "REGIME_STABLE")
        self.assertFalse(mcs.positive_edge(sr))

    def test_non_stable_verdict_is_not_edge(self):
        sr = _sr([0.06, 0.07, 0.05], [0.0, 0.0, 0.0], "CONSISTENT_WITH_NOISE")
        self.assertFalse(mcs.positive_edge(sr))


class TestClassifyCostRobustness(unittest.TestCase):
    def _profile(self, edges):
        # edges: dict lookback -> bool positive edge
        prof = {}
        for lb in mcs.LOOKBACKS:
            if edges.get(str(lb)):
                prof[str(lb)] = _sr([0.06, 0.07], [0.0, 0.0], "REGIME_STABLE")
            else:
                prof[str(lb)] = _sr([0.0, 0.01], [0.0, 0.0], "CONSISTENT_WITH_NOISE")
        return prof

    def test_survives(self):
        zero = self._profile({"3": True, "5": True})
        cost = self._profile({"3": True, "5": True, "10": True})
        out = mcs.classify_cost_robustness(zero, cost)
        self.assertEqual(out["classification"], "SURVIVES")
        self.assertEqual(out["zero_positive_lookbacks"], ["3", "5"])

    def test_eroded(self):
        zero = self._profile({"3": True, "5": True, "10": True})
        cost = self._profile({"3": True, "5": True})
        out = mcs.classify_cost_robustness(zero, cost)
        self.assertEqual(out["classification"], "ERODED")
        self.assertEqual(out["cost_positive_lookbacks"], ["3", "5"])

    def test_no_edge(self):
        zero = self._profile({})
        cost = self._profile({})
        out = mcs.classify_cost_robustness(zero, cost)
        self.assertEqual(out["classification"], "NO_EDGE")


if __name__ == "__main__":
    unittest.main()