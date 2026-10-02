"""Unit tests for regime-stability stress testing.

Deterministic and runnable with `python -m unittest`.
"""
import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def _coin_flip_signals(closes, **params):
    """Adapt coin-flip signals to the (closes, **params) calling convention."""
    return bt.random_signals(len(closes), seed=int(sum(params.values()) * 1000 + 42))


class TestScenarioGeneration(unittest.TestCase):
    """Regime scenarios must be reproducible and diverse."""

    def test_scenarios_reproducible(self):
        s1 = bt.canonical_regime_scenarios(42)
        s2 = bt.canonical_regime_scenarios(42)
        self.assertEqual(len(s1), len(s2))
        for a, b in zip(s1, s2):
            self.assertEqual(a.name, b.name)
            self.assertEqual(a.seed, b.seed)
            self.assertEqual(a.p_transition, b.p_transition)
            for ra, rb in zip(a.regimes, b.regimes):
                self.assertEqual(ra.vol_annual, rb.vol_annual)

    def test_scenarios_diverse(self):
        s = bt.canonical_regime_scenarios(42)
        names = [sc.name for sc in s]
        self.assertEqual(len(set(names)), len(names))  # unique names
        vols = [sc.regimes[0].vol_annual for sc in s]
        self.assertGreater(max(vols), min(vols))  # distinct regime mixes
        seeds = [sc.seed for sc in s]
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_series_from_scenario_is_valid(self):
        s = bt.canonical_regime_scenarios(42)[1]  # turbulent
        bars = bt.generate_under(s, 100)
        self.assertEqual(bars.n_bars, 100)
        self.assertTrue((bars.highs >= bars.lows).all())
        self.assertTrue((bars.closes > 0).all())
        self.assertFalse(np.isnan(bars.closes).any())


class TestRegimeStressDeterminism(unittest.TestCase):
    """A stress result must be fully deterministic given fixed inputs."""

    def test_stress_reproducible(self):
        bars = bt.generate_bars(600, seed=7)

        def signals(closes, fast, slow):
            n = len(closes)
            out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
            for i in range(slow - 1, n):
                fm = np.mean(closes[i - fast + 1 : i + 1])
                sm = np.mean(closes[i - slow + 1 : i + 1])
                out[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
            return out

        grid = [{"fast": f, "slow": s} for f in (10, 20) for s in (30, 60)]
        r1 = bt.regime_stress(
            signals, list(bars), grid, (("fast", 10.0), ("slow", 30.0)),
            train_window=60, test_window=20, warmup=0,
        )
        r2 = bt.regime_stress(
            signals, list(bars), grid, (("fast", 10.0), ("slow", 30.0)),
            train_window=60, test_window=20, warmup=0,
        )
        self.assertEqual(r1.overall_verdict, r2.overall_verdict)
        for a, b in zip(r1.scenarios, r2.scenarios):
            self.assertEqual(a.name, b.name)
            self.assertEqual(a.baseline_median_log_return, b.baseline_median_log_return)
            self.assertEqual(a.edge_status, b.edge_status)
        self.assertEqual(r1.inspect(), r2.inspect())


class TestNoiseBehavior(unittest.TestCase):
    """A coin-flip signal must be consistent with noise in every scenario."""

    def test_noise_consistent_across_regimes(self):
        bars = bt.generate_bars(600, seed=11)
        grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))
        result = bt.regime_stress(
            _coin_flip_signals, list(bars), grid, (("fast", 20.0), ("slow", 60.0)),
            train_window=60, test_window=20, warmup=0,
        )
        self.assertEqual(result.overall_verdict, "CONSISTENT_WITH_NOISE")
        self.assertEqual(result.n_edge_free, len(result.scenarios))
        self.assertEqual(result.n_edge_present, 0)
        for s in result.scenarios:
            self.assertLess(abs(s.baseline_median_log_return - s.noise_median_log_return),
                            bt.OUTLIER_TOL, msg=s.name)

    def test_noise_benchmark_near_zero_per_scenario(self):
        bars = bt.generate_bars(600, seed=13)
        grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))
        result = bt.regime_stress(
            _coin_flip_signals, list(bars), grid, (("fast", 20.0), ("slow", 60.0)),
            train_window=60, test_window=20, warmup=0,
        )
        for s in result.scenarios:
            self.assertLess(abs(s.noise_median_log_return), 0.30, msg=s.name)


class TestDetectsRegimeDependency(unittest.TestCase):
    """A signal whose results swing with the regime family must be flagged."""

    def test_long_only_swing_is_regime_dependent(self):
        bars = bt.generate_bars(600, seed=17)
        grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))
        # A long-only rule on extreme up/neutral/down regimes:
        # large edge in up, ~noise in neutral, large drag in down.
        scenarios = bt.stress_regime_scenarios()
        result = bt.regime_stress(
            bt.always_long_signal, list(bars), grid,
            (("fast", 20.0), ("slow", 60.0)),
            scenarios=scenarios,
            train_window=60, test_window=20, warmup=0,
        )
        self.assertEqual(result.overall_verdict, "REGIME_DEPENDENT")
        self.assertTrue(result.regime_dependent)
        # The candidate medians swing from large positive to large negative,
        # far more than the coin-flip null varies across scenarios.
        self.assertGreater(result.candidate_dispersion, 0.05)
        self.assertGreater(result.candidate_dispersion, 2.0 * result.null_dispersion)

    def test_adaptive_direction_is_regime_stable(self):
        bars = bt.generate_bars(600, seed=19)
        grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))
        # Direction-following rule: long in up regimes, short in down
        # regimes; both produce a positive edge, i.e. stable behavior.
        stress = bt.stress_regime_scenarios()
        up_down = [s for s in stress if s.name in ("strong_up", "strong_down")]
        result = bt.regime_stress(
            bt.direction_signal, list(bars), grid,
            (("fast", 20.0), ("slow", 60.0)),
            scenarios=up_down,
            train_window=60, test_window=20, warmup=0,
        )
        self.assertEqual(result.overall_verdict, "REGIME_STABLE")
        self.assertFalse(result.regime_dependent)
        # The adaptive signal earns a positive edge in both regimes
        # (long in up, short in down); the cross-scenario dispersion is
        # small, i.e. the behavior is stable.
        self.assertGreater(result.n_edge_present, 0)
        self.assertLess(result.candidate_dispersion, 2.0 * result.null_dispersion)


class TestRegimeStressEdgeCases(unittest.TestCase):
    """Invalid inputs must fail early and predictably."""

    def test_empty_bars_raises(self):
        with self.assertRaises(AssertionError):
            bt.regime_stress(
                lambda closes, **p: [bt.Signal(date=1, weight=0.0)],
                bt.generate_bars(0, seed=23),
                [{"x": 1}],
                (("x", 1.0),),
                train_window=10, test_window=5, warmup=0,
            )

    def test_signal_length_mismatch_raises(self):
        bars = bt.generate_bars(200, seed=29)

        def short_signals(closes, **p):
            return [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes) - 10)]

        grid = [{"x": 1}]
        with self.assertRaises(ValueError):
            bt.regime_stress(
                short_signals, list(bars), grid, (("x", 1.0),),
                train_window=30, test_window=20, warmup=0,
            )

    def test_short_series_under_a_scenario(self):
        # With a short series, the conditional signal's early-vol window
        # degrades gracefully to neutral rather than crashing.
        bars = bt.generate_bars(50, seed=31)
        cond = bt.conditional_signal(bt.deterministic_edge_signal, bt.active_when_turbulent)
        result = bt.regime_stress(
            cond, list(bars), [{"fast": 10.0, "slow": 30.0}],
            (("fast", 10.0), ("slow", 30.0)),
            train_window=10, test_window=5, warmup=0,
        )
        self.assertIn(result.overall_verdict, ("CONSISTENT_WITH_NOISE", "REGIME_DEPENDENT"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
