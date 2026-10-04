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


def _make_two_regime_bars(n_bars: int = 800, seed: int = 99) -> bt.BarSequence:
    """Build a deterministic bar series with two volatility regimes.

    Segment 1 (calm): drift 0 / low vol -- a long-only position is ~noise.
    Segment 2 (up-drift): strong positive drift -- a long-only position has a
    large edge. volatility_segments separates them, so always_long_signal is
    regime-dependent by construction. Used as a known-vertex for
    stress_segments.
    """
    regimes = [
        bt.Regime(drift_annual=0.0, vol_annual=0.02, intraday_range_scale=0.005),
        bt.Regime(drift_annual=1.5, vol_annual=0.15, intraday_range_scale=0.02),
    ]
    return bt.generate_bars(
        n_bars=n_bars, regimes=regimes, p_transition=0.0,
        start_price=100.0, seed=seed,
    )


def _make_aapl_fixture():
    """Load the AAPL collected series, its past-only volatility labels, and MA signals once."""
    from research.data.preflight import load_manifest

    manifest = load_manifest()
    assert manifest["dataset_id"].startswith("yf-ohlcv"), "unexpected manifest"
    bars, dates = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    labels = bt.volatility_segments(closes, window=60)
    seg_fn = bt.segment_fn_from_labels(labels)
    # MA-crossover signals matching the canonical (20, 60) baseline; past-only.
    fast, slow = 20, 60
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes))]
    for i in range(slow - 1, len(closes)):
        fm = np.mean(closes[i - fast + 1 : i + 1])
        sm = np.mean(closes[i - slow + 1 : i + 1])
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
    return bars, seg_fn, signals


class TestStressSegmentsConstruction(unittest.TestCase):
    """Segmentation of a series by a past-only segment function."""

    def test_segment_fn_from_labels_contract(self):
        labels = ["insufficient"] * 40 + ["calm"] * 200 + ["turbulent"] * 300
        seg_fn = bt.segment_fn_from_labels(labels)
        for i, exp in enumerate(labels):
            self.assertEqual(seg_fn(np.zeros(1000), i), exp)

    def test_volatility_segments_past_only_prefix(self):
        closes = np.ones(100) * 100.0
        labels = bt.volatility_segments(closes, window=30)
        self.assertEqual(sum(1 for l in labels if l == "insufficient"), 30)

    def test_stress_segments_segment_boundaries(self):
        bars = _make_two_regime_bars(600, seed=99)
        closes = bars.closes_array()
        labels = bt.volatility_segments(closes, window=30)
        seg_fn = bt.segment_fn_from_labels(labels)
        baseline = (("fast", 20.0), ("slow", 60.0))
        grid = [{"fast": 20.0, "slow": 60.0}]
        neutral_signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes))]

        def neutral_signals_fn(closes, **p):
            return [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes))]

        stress = bt.stress_segments(
            signals_fn=neutral_signals_fn,
            bars=bars,
            signals=neutral_signals,
            param_grid=grid,
            baseline=baseline,
            segment_fn=seg_fn,
            train_window=60, test_window=20,
            warmup=0, min_segment_bars=200,
        )
        # leading "insufficient" prefix is dropped; calm + normal -> 2 segments
        self.assertEqual(len(stress.scenarios), 2)
        names = [s.name for s in stress.scenarios]
        self.assertIn("calm", names)
        for s in stress.scenarios:
            self.assertGreater(s.n_folds, 0)
            self.assertGreater(s.n_periods, 0)
            self.assertTrue(
                np.isfinite(s.baseline_median_log_return) and np.isfinite(s.noise_median_log_return),
                msg=s.name,
            )

    def test_stress_segments_min_segment_filtering(self):
        bars = bt.generate_bars(860, seed=53)
        closes = bars.closes_array()
        labels = (["calm"] * 400) + (["normal"] * 60) + (["turbulent"] * 400)
        seg_fn = bt.segment_fn_from_labels(labels)
        grid = [{"fast": 20.0, "slow": 60.0}]
        baseline = (("fast", 20.0), ("slow", 60.0))
        neutral_signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes))]

        def neutral_signals_fn(closes, **p):
            return [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes))]

        stress_lo = bt.stress_segments(
            signals_fn=neutral_signals_fn,
            bars=bars,
            signals=neutral_signals,
            param_grid=grid,
            baseline=baseline,
            segment_fn=seg_fn,
            train_window=60, test_window=20, min_segment_bars=50,
        )
        stress_hi = bt.stress_segments(
            signals_fn=neutral_signals_fn,
            bars=bars,
            signals=neutral_signals,
            param_grid=grid,
            baseline=baseline,
            segment_fn=seg_fn,
            train_window=60, test_window=20, min_segment_bars=250,
        )
        self.assertEqual(len(stress_lo.scenarios), 3)
        self.assertEqual(len(stress_hi.scenarios), 2)

    def test_stress_segments_short_segments_raise(self):
        bars = bt.generate_bars(150, seed=61)
        labels = ["calm"] * 150
        seg_fn = bt.segment_fn_from_labels(labels)
        grid = [{"fast": 20.0, "slow": 60.0}]
        baseline = (("fast", 20.0), ("slow", 60.0))
        neutral_signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(len(bars))]
        with self.assertRaises(ValueError):
            bt.stress_segments(
                lambda c, **p: neutral_signals,
                bars=bars,
                signals=neutral_signals,
                param_grid=grid,
                baseline=baseline,
                segment_fn=seg_fn,
                train_window=60, test_window=20, min_segment_bars=400,
            )

    def test_stress_segments_signal_length_mismatch_raises(self):
        bars = bt.generate_bars(400, seed=67)
        labels = (["calm"] * 200) + (["turbulent"] * 200)
        seg_fn = bt.segment_fn_from_labels(labels)
        grid = [{"fast": 20.0, "slow": 60.0}]
        baseline = (("fast", 20.0), ("slow", 60.0))
        # The signals_fn returns fewer signals than there are bars; the sweep
        # then rejects it as a length mismatch. The `signals` kwarg is correct
        # length here, so it does not mask the mismatch.
        def short_signals_fn(closes, **p):
            return [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes) - 5)]
        with self.assertRaises(ValueError):
            bt.stress_segments(
                signals_fn=short_signals_fn,
                bars=bars,
                signals=[bt.Signal(date=i + 1, weight=0.0) for i in range(len(bars))],
                param_grid=grid,
                baseline=baseline,
                segment_fn=seg_fn,
                train_window=60, test_window=20, min_segment_bars=50,
            )


class TestStressSegmentsDeterminism(unittest.TestCase):
    """A stress_segments result must be fully deterministic given fixed inputs."""

    def test_stress_segments_determinism(self):
        bars, seg_fn, signals = _make_aapl_fixture()
        baseline = (("fast", 20.0), ("slow", 60.0))
        grid = [{"fast": 20.0, "slow": 60.0}]

        def signals_fn(closes, fast, slow):
            fast, slow = int(fast), int(slow)
            n = len(closes)
            out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
            for i in range(slow - 1, n):
                fm = np.mean(closes[i - fast + 1 : i + 1])
                sm = np.mean(closes[i - slow + 1 : i + 1])
                out[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
            return out

        r1 = bt.stress_segments(
            signals_fn=signals_fn,
            bars=bars,
            signals=signals,
            param_grid=grid,
            baseline=baseline,
            segment_fn=seg_fn,
            train_window=60, test_window=20, warmup=0, min_segment_bars=200,
        )
        r2 = bt.stress_segments(
            signals_fn=signals_fn,
            bars=bars,
            signals=signals,
            param_grid=grid,
            baseline=baseline,
            segment_fn=seg_fn,
            train_window=60, test_window=20, warmup=0, min_segment_bars=200,
        )
        self.assertEqual(r1.overall_verdict, r2.overall_verdict)
        for a, b in zip(r1.scenarios, r2.scenarios):
            self.assertEqual(a.name, b.name)
            self.assertEqual(a.n_folds, b.n_folds)
            self.assertEqual(a.n_periods, b.n_periods)
            self.assertEqual(a.baseline_median_log_return, b.baseline_median_log_return)
            self.assertEqual(a.noise_median_log_return, b.noise_median_log_return)
            self.assertEqual(a.edge_status, b.edge_status)
        self.assertEqual(r1.inspect(), r2.inspect())


class TestStressSegmentsKnownVertex(unittest.TestCase):
    """Regime-dependence on real structure must be detectable."""

    def test_always_long_across_calm_and_up_drift_is_regime_dependent(self):
        bars = _make_two_regime_bars(800, seed=99)
        closes = bars.closes_array()
        labels = bt.volatility_segments(closes, window=60)
        seg_fn = bt.segment_fn_from_labels(labels)
        grid = [{"fast": 20.0, "slow": 60.0}]
        baseline = (("fast", 20.0), ("slow", 60.0))
        long_signals = [bt.Signal(date=i + 1, weight=1.0) for i in range(len(closes))]
        result = bt.stress_segments(
            signals_fn=bt.always_long_signal,
            bars=bars,
            signals=long_signals,
            param_grid=grid,
            baseline=baseline,
            segment_fn=seg_fn,
            train_window=60, test_window=20, warmup=0, min_segment_bars=200,
        )
        self.assertEqual(result.overall_verdict, "REGIME_DEPENDENT")
        self.assertTrue(result.regime_dependent)
        # Candidate medians swing from ~noise (calm) to a large edge (up-drift);
        # the swing must exceed the coin-flip null's swing across the same
        # segments.
        self.assertGreater(result.candidate_dispersion, 0.03)
        self.assertGreater(result.candidate_dispersion, result.null_dispersion)

    def test_stress_segments_coin_flip_stays_bounded_across_real_segments(self):
        bars, seg_fn, signals = _make_aapl_fixture()
        grid = [{"fast": 20.0, "slow": 60.0}]
        baseline = (("fast", 20.0), ("slow", 60.0))

        result = bt.stress_segments(
            signals_fn=lambda c, **p: [bt.Signal(date=i + 1, weight=0.0) for i in range(len(c))],
            bars=bars,
            signals=signals,
            param_grid=grid,
            baseline=baseline,
            segment_fn=seg_fn,
            train_window=60, test_window=20, warmup=0, min_segment_bars=200,
        )
        # A coin-flip signal cannot have regime dependence: its dispersion
        # across segments stays small, so it never lands REGIME_DEPENDENT.
        self.assertFalse(result.regime_dependent)
        self.assertLess(result.candidate_dispersion, 0.35)
        self.assertLess(result.null_dispersion, 0.35)


class TestStressSegmentsRealData(unittest.TestCase):
    """A run on the collected universe must produce a well-formed result."""

    def test_stress_segments_real_data_structure(self):
        bars, seg_fn, signals = _make_aapl_fixture()
        baseline = (("fast", 20.0), ("slow", 60.0))
        grid = [{"fast": 20.0, "slow": 60.0}]

        def signals_fn(closes, fast, slow):
            fast, slow = int(fast), int(slow)
            n = len(closes)
            out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
            for i in range(slow - 1, n):
                fm = np.mean(closes[i - fast + 1 : i + 1])
                sm = np.mean(closes[i - slow + 1 : i + 1])
                out[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
            return out

        result = bt.stress_segments(
            signals_fn=signals_fn,
            bars=bars,
            signals=signals,
            param_grid=grid,
            baseline=baseline,
            segment_fn=seg_fn,
            train_window=60, test_window=20, warmup=0, min_segment_bars=200,
        )
        self.assertGreater(len(result.scenarios), 1)
        self.assertIn(result.overall_verdict,
                      ("REGIME_STABLE", "REGIME_DEPENDENT", "CONSISTENT_WITH_NOISE"))
        for s in result.scenarios:
            self.assertGreater(s.n_folds, 0)
            self.assertGreater(s.n_periods, 0)
            self.assertTrue(np.isfinite(s.baseline_median_log_return))
            self.assertTrue(np.isfinite(s.noise_median_log_return))
        self.assertIsInstance(result.inspect(), str)
        self.assertGreater(len(result.inspect()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
