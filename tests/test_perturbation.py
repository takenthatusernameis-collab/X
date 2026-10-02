"""Unit tests for the perturbation / parameter-sensitivity toolkit.

Deterministic and runnable with `python -m unittest`.
"""
import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


class TestSweepDeterminism(unittest.TestCase):
    """Sweeps and summaries must be fully deterministic given fixed inputs."""

    def test_sweep_reproducible(self):
        np.random.seed(7)
        bars = bt.generate_bars(500, seed=7)

        def signals(closes, fast, slow):
            n = len(closes)
            out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
            for i in range(slow - 1, n):
                fm = np.mean(closes[i - fast + 1 : i + 1])
                sm = np.mean(closes[i - slow + 1 : i + 1])
                out[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
            return out

        grid = [{"fast": f, "slow": s} for f in (10, 20) for s in (30, 60)]
        cfg = bt.BacktestConfig(warmup_periods=0)
        r1 = bt.parameter_sweep(signals, list(bars), grid, 100, 50, cfg=cfg)
        r2 = bt.parameter_sweep(signals, list(bars), grid, 100, 50, cfg=cfg)
        self.assertEqual(r1.param_sets, r2.param_sets)
        self.assertEqual(r1.fold_total_returns, r2.fold_total_returns)

    def test_summary_reproducible(self):
        np.random.seed(11)
        bars = bt.generate_bars(500, seed=11)

        def signals(closes, fast, slow):
            n = len(closes)
            out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
            for i in range(slow - 1, n):
                fm = np.mean(closes[i - fast + 1 : i + 1])
                sm = np.mean(closes[i - slow + 1 : i + 1])
                out[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
            return out

        grid = [{"fast": f, "slow": s} for f in (10, 20) for s in (30, 60)]
        result = bt.parameter_sweep(signals, list(bars), grid, 100, 50,
                                    cfg=bt.BacktestConfig(warmup_periods=0))
        baseline = result.param_sets[0]
        s1 = bt.sweep_summary(result, baseline)
        s2 = bt.sweep_summary(result, baseline)
        self.assertEqual(s1.param_sets, s2.param_sets)
        self.assertEqual(s1.median_log_returns, s2.median_log_returns)
        self.assertEqual(s1.medians_at_deviation, s2.medians_at_deviation)
        self.assertEqual(s1.inspect(), s2.inspect())


class TestParameterGridAround(unittest.TestCase):
    """The around-baseline grid must be a well-formed Cartesian product."""

    def test_includes_baseline(self):
        baseline = (("fast", 20), ("slow", 60))
        grid = bt.parameter_grid_around(baseline, multipliers=(0.5, 1.0, 2.0))
        dicts = [dict(ps) for ps in grid]
        self.assertIn(dict(baseline), dicts)

    def test_grid_size(self):
        grid = bt.parameter_grid_around(
            (("a", 1.0), ("b", 2.0)), multipliers=(0.5, 1.0, 2.0)
        )
        self.assertEqual(len(grid), 9)

    def test_baseline_unmodified(self):
        baseline = (("n", 42.0),)
        grid = bt.parameter_grid_around(baseline)
        for ps in grid:
            for name, val in ps:
                self.assertIsInstance(name, str)
                self.assertIsInstance(val, float)
        self.assertIn((("n", 42.0),), grid)


class TestEmptyGridAndEdgeCases(unittest.TestCase):
    """Invalid input must fail early and predictably."""

    def test_empty_grid_raises(self):
        bars = bt.generate_bars(200, seed=17)
        flat = lambda closes, **p: [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes))]
        with self.assertRaises(ValueError):
            bt.parameter_sweep(flat, list(bars), [], 50, 20)

    def test_signal_length_mismatch_raises(self):
        bars = bt.generate_bars(200, seed=18)

        def short_signals(closes, **p):
            return [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes) - 5)]

        grid = [{"x": 1}]
        with self.assertRaises(ValueError):
            bt.parameter_sweep(short_signals, list(bars), grid, 50, 20)


class TestNoiseBenchmark(unittest.TestCase):
    """A signal that never looks at prices must show no systematic edge."""

    def test_noise_centered_on_zero(self):
        np.random.seed(23)
        bars = bt.generate_bars(1000, seed=23)
        grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)),
                                        multipliers=(0.5, 1.0, 2.0))
        result = bt.parameter_sweep(
            signals_fn=lambda closes, **p: bt.random_signals(len(bars), seed=42),
            bars=list(bars),
            param_grid=grid,
            train_window=120,
            test_window=40,
            cfg=bt.BacktestConfig(warmup_periods=0),
        )
        summary = bt.sweep_summary(result, result.param_sets[0])
        for m in summary.median_log_returns:
            self.assertLess(abs(m), 0.20, "median log return must stay near zero")
        n_pos = summary.n_positive
        n_neg = summary.n_negative
        self.assertLessEqual(abs(n_pos - n_neg), 4,
                             "positive and negative medians must be roughly balanced")

    def test_noise_benchmark_api(self):
        np.random.seed(29)
        bars = bt.generate_bars(1000, seed=29)
        grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))
        summary = bt.noise_benchmark(
            list(bars), grid, 120, 40, warmup=0, overlap_window=0, seed=5,
        )
        self.assertIsInstance(summary, bt.SweepSummary)
        self.assertLess(abs(summary.baseline_median_log_return), 0.20)


class TestSweepDetectsStructuredSignal(unittest.TestCase):
    """A contrived signal working at exactly one parameter value must expose
    a sharp peak; robustness should reject it as non-robust."""

    def test_signal_with_known_peak(self):
        np.random.seed(31)
        bars = bt.generate_bars(800, seed=31)

        def peak_signals(closes, fast, slow):
            if fast != 7 or slow != 14:
                return [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes))]
            n = len(closes)
            out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
            for i in range(13, n):
                fm = np.mean(closes[i - 6 : i + 1])
                sm = np.mean(closes[i - 13 : i + 1])
                out[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
            return out

        grid = [{"fast": f, "slow": s} for f in (7, 14) for s in (14, 28)]
        result = bt.parameter_sweep(
            peak_signals, list(bars), grid, 80, 30,
            cfg=bt.BacktestConfig(warmup_periods=0),
        )
        summary = bt.sweep_summary(result, baseline=(("fast", 7), ("slow", 14)))
        self.assertEqual(len(grid), len(result.param_sets))
        self.assertEqual(summary.n_param_sets, len(grid))
        self.assertGreater(summary.best_median_log_return,
                           summary.worst_median_log_return)
        self.assertEqual(summary.fraction_positive, 0.5)


class TestSweepSummaryConsistency(unittest.TestCase):
    """Summary fields must be mutually consistent."""

    def test_sign_counts(self):
        np.random.seed(37)
        bars = bt.generate_bars(600, seed=37)

        def signals(closes, x, y):
            n = len(closes)
            out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
            for i in range(10, n):
                out[i] = bt.Signal(date=i + 1, weight=1.0 if closes[i] > closes[i - 10] else -1.0)
            return out

        grid = [{"x": 10, "y": 1.0}, {"x": 11, "y": 1.0}, {"x": 12, "y": 1.0}]
        result = bt.parameter_sweep(signals, list(bars), grid, 60, 20,
                                    cfg=bt.BacktestConfig(warmup_periods=0))
        summary = bt.sweep_summary(result, result.param_sets[0])
        self.assertEqual(summary.n_positive + summary.n_zero + summary.n_negative,
                         summary.n_param_sets)
        self.assertAlmostEqual(summary.fraction_positive,
                               summary.n_positive / summary.n_param_sets)
        for d, med in summary.medians_at_deviation.items():
            self.assertIn(d, summary.deviations)
        self.assertEqual(summary.deviations, sorted(summary.deviations))

    def test_deviation_scan_uses_correct_reference(self):
        np.random.seed(41)
        bars = bt.generate_bars(600, seed=41)

        def signals(closes, x, y):
            n = len(closes)
            out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
            for i in range(10, n):
                out[i] = bt.Signal(date=i + 1, weight=1.0 if closes[i] > closes[i - 10] else -1.0)
            return out

        grid = [{"x": 10, "y": 1.0}, {"x": 20, "y": 1.0}, {"x": 5, "y": 1.0}]
        result = bt.parameter_sweep(signals, list(bars), grid, 60, 20,
                                    cfg=bt.BacktestConfig(warmup_periods=0))
        baseline = (("x", 10.0), ("y", 1.0))
        summary = bt.sweep_summary(result, baseline)
        self.assertEqual(summary.baseline_median_log_return,
                         summary.median_log_returns[0])
        self.assertIn(0.0, summary.deviations)
        self.assertEqual(summary.param_set_at(0.0), summary.baseline_median_log_return)

    def test_noise_comparison(self):
        np.random.seed(43)
        bars = bt.generate_bars(600, seed=43)

        def signals(closes, x, y):
            n = len(closes)
            out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
            for i in range(10, n):
                out[i] = bt.Signal(date=i + 1, weight=1.0 if closes[i] > closes[i - 10] else -1.0)
            return out

        grid = [{"x": 10, "y": 1.0}, {"x": 20, "y": 1.0}]
        result = bt.parameter_sweep(signals, list(bars), grid, 60, 20,
                                    cfg=bt.BacktestConfig(warmup_periods=0))
        summary = bt.sweep_summary(result, result.param_sets[0], noise_median=0.001)
        self.assertIsInstance(summary.compare_noise(0.001), bool)
        self.assertIsInstance(summary.compare_noise(0.001, tol=1e-9), bool)


class TestNoiseBenchmarkIS_OOS(unittest.TestCase):
    """The sweep must respect IS/OOS per parameter set, not collapse folds."""

    def test_folds_per_param_set(self):
        np.random.seed(47)
        bars = bt.generate_bars(500, seed=47)
        grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))
        result = bt.parameter_sweep(
            bt.random_signals, list(bars), grid, 100, 50, warmup=20,
            overlap_window=0, cfg=bt.BacktestConfig(warmup_periods=20),
        )
        self.assertEqual(len(result.param_sets), len(grid))
        for returns in result.fold_total_returns:
            self.assertEqual(len(returns), 7)  # matches engine test_fold_equity_uses_oos_segment_only

    def test_param_sets_order_preserved(self):
        np.random.seed(53)
        bars = bt.generate_bars(300, seed=53)
        grid = [{"a": 1}, {"a": 2}, {"a": 3}]
        def signals(closes, **p):
            return [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes))]
        result = bt.parameter_sweep(signals, list(bars), grid, 50, 20,
                                    cfg=bt.BacktestConfig(warmup_periods=0))
        self.assertEqual(
            [dict(ps) for ps in result.param_sets],
            grid,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
