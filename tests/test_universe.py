"""Unit tests for the asset-universe robustness toolkit.

Deterministic (seeded) and runnable with `python -m unittest`.
"""
import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from examples.ma_crossover import ma_crossover_signals


class TestUniverseReproducibility(unittest.TestCase):
    """Sweep output must be deterministic."""

    def test_sweep_reproducible(self):
        """Two sweeps with the same seed must produce byte-identical medians."""
        np.random.seed(42)
        assets = bt.uniform_regime_assets(
            [
                bt.Regime(drift_annual=0.03, vol_annual=0.18, intraday_range_scale=0.015),
                bt.Regime(drift_annual=-0.01, vol_annual=0.35, intraday_range_scale=0.030),
            ],
            n_assets=5,
            drift_offsets=[-0.03, 0.0, 0.03, 0.06, 0.09],
            seed=7,
            n_bars=800,
        )
        grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))

        result1 = bt.sweep_across_assets(
            signals_fn=ma_crossover_signals,
            assets=assets,
            param_grid=grid,
            baseline=(("fast", 20), ("slow", 60)),
            train_window=120,
            test_window=40,
            warmup=30,
            overlap_window=20,
        )
        summary1 = bt.asset_sweep_summary(result1, baseline=(("fast", 20), ("slow", 60)))

        result2 = bt.sweep_across_assets(
            signals_fn=ma_crossover_signals,
            assets=assets,
            param_grid=grid,
            baseline=(("fast", 20), ("slow", 60)),
            train_window=120,
            test_window=40,
            warmup=30,
            overlap_window=20,
        )
        summary2 = bt.asset_sweep_summary(result2, baseline=(("fast", 20), ("slow", 60)))

        self.assertEqual(result1.baseline_median_log_returns,
                         result2.baseline_median_log_returns)
        self.assertEqual(summary1.median_log_returns, summary2.median_log_returns)
        self.assertEqual(summary1.assets_positive_share, summary2.assets_positive_share)
        self.assertEqual(summary1.verdict, summary2.verdict)

        # The default is now per_asset_null=True: the default path must
        # report per-asset significance flags (documented contract).
        self.assertIsNotNone(summary1.asset_null_medians)
        self.assertIsNotNone(summary1.asset_significance)
        self.assertEqual(len(summary1.asset_null_medians), summary1.n_assets)

    def test_asset_family_reproducible(self):
        """uniform_regime_assets must regenerate the same family each call."""
        a1 = bt.uniform_regime_assets(
            [bt.Regime(drift_annual=0.05, vol_annual=0.20)],
            n_assets=3,
            drift_offsets=[0.0, 0.05, 0.10],
            seed=11,
            n_bars=300,
        )
        a2 = bt.uniform_regime_assets(
            [bt.Regime(drift_annual=0.05, vol_annual=0.20)],
            n_assets=3,
            drift_offsets=[0.0, 0.05, 0.10],
            seed=11,
            n_bars=300,
        )
        for b1, b2 in zip(a1, a2):
            self.assertTrue((b1.closes_array() == b2.closes_array()).all())


class TestUniverseEdgeCases(unittest.TestCase):
    """Empty and mismatched inputs must fail loudly."""

    def test_empty_assets_raises(self):
        """An empty asset list must raise."""
        with self.assertRaises(ValueError):
            bt.sweep_across_assets(
                signals_fn=lambda closes, **p: [],
                assets=[],
                param_grid=[{"fast": 20}],
                baseline=(("fast", 20),),
                train_window=50,
                test_window=20,
            )

    def test_asset_names_mismatch_raises(self):
        """asset_names length must match n_assets."""
        assets = bt.flat_regime_assets(3)
        with self.assertRaises(ValueError):
            bt.sweep_across_assets(
                signals_fn=lambda closes, **p: [bt.Signal(date=i + 1, weight=0.0)
                                                for i in range(len(closes))],
                assets=assets,
                param_grid=[{"fast": 20}],
                baseline=(("fast", 20),),
                train_window=50,
                test_window=20,
                asset_names=["one"],
            )

    def test_single_asset_edge_case(self):
        """A single asset must run without error and return a verdict."""
        assets = bt.flat_regime_assets(1, seed=42)
        grid = [bt.parameter_grid_around((("fast", 20), ("slow", 60)))[0]]
        result = bt.sweep_across_assets(
            signals_fn=ma_crossover_signals,
            assets=assets,
            param_grid=grid,
            baseline=(("fast", 20), ("slow", 60)),
            train_window=60,
            test_window=20,
            warmup=10,
        )
        summary = bt.asset_sweep_summary(result, baseline=(("fast", 20), ("slow", 60)))
        self.assertEqual(summary.n_assets, 1)
        self.assertIn(summary.verdict, (bt.ASSET_VERDICT_CONCENTRATED,
                                         bt.ASSET_VERDICT_NO_EDGE))
        self.assertEqual(summary.asset_names, ["asset_0"])


class TestConcentrationDetection(unittest.TestCase):
    """The sweep must detect when wins cluster in a single asset."""

    def test_concentrated_detection(self):
        """One strong asset + several flat ones must be flagged CONCENTRATED."""
        np.random.seed(42)
        base = [
            bt.Regime(drift_annual=0.02, vol_annual=0.20),
            bt.Regime(drift_annual=-0.01, vol_annual=0.30),
        ]
        # The last asset has a very strong positive drift; the others have
        # strong negative drift. always_long wins almost everywhere in the
        # strong asset and almost nowhere elsewhere: wins must be concentrated.
        offsets = [0.90, -0.30, -0.30, -0.30, -0.30, -0.30]
        assets = bt.uniform_regime_assets(base, n_assets=6, drift_offsets=offsets,
                                          seed=13, n_bars=800)

        def always_long(closes, **p):
            return [bt.Signal(date=i + 1, weight=1.0) for i in range(len(closes))]

        result = bt.sweep_across_assets(
            signals_fn=always_long,
            assets=assets,
            param_grid=[{"fast": 20}],
            baseline=(("fast", 20),),
            train_window=120,
            test_window=40,
            warmup=0,
        )
        summary = bt.asset_sweep_summary(result, baseline=(("fast", 20),))
        self.assertEqual(summary.verdict, bt.ASSET_VERDICT_CONCENTRATED)
        self.assertGreater(summary.best_asset_concentration, 0.5)
        self.assertEqual(summary.best_asset, "asset_0")

    def test_consistent_detection(self):
        """Strong uniform positive drifts must yield CONSISTENT (edge present
        in every asset, wins spread, no winner-take-all)."""
        np.random.seed(42)
        base = [
            bt.Regime(drift_annual=0.02, vol_annual=0.14),
            bt.Regime(drift_annual=0.01, vol_annual=0.16),
        ]
        # Strong, tight, uniformly positive drifts: every asset earns a
        # clear edge well above the coin-flip null dispersion.
        offsets = [0.40, 0.41, 0.42, 0.43, 0.44, 0.45]
        assets = bt.uniform_regime_assets(base, n_assets=6, drift_offsets=offsets,
                                          seed=17, n_bars=800)

        def always_long(closes, **p):
            return [bt.Signal(date=i + 1, weight=1.0) for i in range(len(closes))]

        result = bt.sweep_across_assets(
            signals_fn=always_long,
            assets=assets,
            param_grid=[{"fast": 20}],
            baseline=(("fast", 20),),
            train_window=120,
            test_window=40,
            warmup=0,
        )
        summary = bt.asset_sweep_summary(result, baseline=(("fast", 20),))
        self.assertEqual(summary.verdict, bt.ASSET_VERDICT_CONSISTENT)
        self.assertLess(summary.assets_positive_share, 0.60)
        # Edge present in all assets (all drifts strongly positive)
        self.assertEqual(summary.positive_assets, 6)


class TestNoEdgeDetection(unittest.TestCase):
    """A strategy with no edge must be flagged NO_EDGE against the null."""

    def test_no_edge_on_flat_family(self):
        """MA crossover on a flat, mean-reverting family must be noise-like."""
        assets = bt.flat_regime_assets(6, seed=42, n_bars=800)
        grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))

        result = bt.sweep_across_assets(
            signals_fn=ma_crossover_signals,
            assets=assets,
            param_grid=grid,
            baseline=(("fast", 20), ("slow", 60)),
            train_window=120,
            test_window=40,
            warmup=30,
            overlap_window=20,
        )
        summary = bt.asset_sweep_summary(result, baseline=(("fast", 20), ("slow", 60)))
        self.assertEqual(summary.verdict, bt.ASSET_VERDICT_NO_EDGE)

    def test_noise_like_medians_near_null(self):
        """NO_EDGE medians must sit within tolerance of the coin-flip null."""
        assets = bt.flat_regime_assets(4, seed=99, n_bars=800)
        grid = [{"fast": 20}]
        result = bt.sweep_across_assets(
            signals_fn=lambda closes, **p: [bt.Signal(date=i + 1, weight=0.0)
                                            for i in range(len(closes))],
            assets=assets,
            param_grid=grid,
            baseline=(("fast", 20),),
            train_window=120,
            test_window=40,
            warmup=10,
        )
        summary = bt.asset_sweep_summary(result, baseline=(("fast", 20),))
        self.assertEqual(summary.verdict, bt.ASSET_VERDICT_NO_EDGE)
        for med in summary.median_log_returns:
            self.assertLessEqual(abs(med - summary.null_median_log_return), 0.05)


class TestSummaryConcentrationMath(unittest.TestCase):
    """Concentration math must be verifiable from a hand-constructed result."""

    def test_positive_share_computation(self):
        """assets_positive_share = best asset wins / total wins."""
        # 3 assets, 2 param sets, 2 folds each:
        # asset 0: 1 win, asset 1: 0 wins, asset 2: 4 wins -> total 5, share 80%.
        fold_rets = [
            [[0.01, -0.01], [0.00, -0.02]],  # asset 0: 1 win
            [[0.00, -0.01], [0.00, -0.03]],  # asset 1: 0 wins
            [[0.04, 0.05], [0.01, 0.02]],    # asset 2: 4 wins
        ]
        result = bt.AssetSweepResult(
            assets=[bt.generate_bars(200, seed=i) for i in range(3)],
            asset_names=["a", "b", "c"],
            param_sets=[("fast", 20), ("fast", 40)],
            fold_total_returns=fold_rets,
            # Edge clearly present in b and c, absent in a (null sits near 0).
            baseline_median_log_returns=[-0.02, 0.03, 0.10],
            n_folds=2,
            n_periods=200,
            train_window_bars=120,
            test_window_bars=40,
        )
        summary = bt.asset_sweep_summary(result, baseline=(("fast", 20),))
        self.assertEqual(summary.best_asset, "c")
        self.assertAlmostEqual(summary.assets_positive_share, 10.0 / 13.0)
        # 80% >= threshold -> CONCENTRATED
        self.assertEqual(summary.verdict, bt.ASSET_VERDICT_CONCENTRATED)


class TestPerAssetNull(unittest.TestCase):
    """Per-asset coin-flip null and significance flags."""

    def test_per_asset_null_reproducible(self):
        """Per-asset significance must be fully deterministic."""
        np.random.seed(42)
        assets = bt.uniform_regime_assets(
            [
                bt.Regime(drift_annual=0.03, vol_annual=0.18),
                bt.Regime(drift_annual=-0.01, vol_annual=0.35),
            ],
            n_assets=4, drift_offsets=[-0.06, 0.0, 0.03, 0.12], seed=7, n_bars=800,
        )
        grid = [bt.parameter_grid_around((("fast", 20), ("slow", 60)))[0]]
        result = bt.sweep_across_assets(
            ma_crossover_signals, assets, grid, (("fast", 20), ("slow", 60)),
            train_window=120, test_window=40, warmup=30,
        )
        s1 = bt.asset_sweep_summary(result, (("fast", 20), ("slow", 60)),
                                    per_asset_null=True)
        s2 = bt.asset_sweep_summary(result, (("fast", 20), ("slow", 60)),
                                    per_asset_null=True)
        self.assertEqual(s1.asset_null_medians, s2.asset_null_medians)
        self.assertEqual(s1.asset_significance, s2.asset_significance)
        self.assertEqual(s1.verdict, s2.verdict)

    def test_per_asset_null_flat_family_no_edge(self):
        """A flat family must still be NO_EDGE under the per-asset null."""
        assets = bt.flat_regime_assets(6, seed=42, n_bars=800)
        grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))
        result = bt.sweep_across_assets(
            ma_crossover_signals, assets, grid, (("fast", 20), ("slow", 60)),
            train_window=120, test_window=40, warmup=30,
        )
        summary = bt.asset_sweep_summary(result, (("fast", 20), ("slow", 60)),
                                        per_asset_null=True)
        self.assertEqual(summary.verdict, bt.ASSET_VERDICT_NO_EDGE)
        self.assertIsNotNone(summary.asset_null_medians)
        self.assertIsNotNone(summary.asset_significance)
        self.assertEqual(sum(summary.asset_significance), 0)

    def test_per_asset_null_agrees_with_global_on_flat(self):
        """Per-asset null must not change a clear NO_EDGE verdict."""
        assets = bt.flat_regime_assets(6, seed=42, n_bars=800)
        grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))
        result = bt.sweep_across_assets(
            ma_crossover_signals, assets, grid, (("fast", 20), ("slow", 60)),
            train_window=120, test_window=40, warmup=30,
        )
        s_global = bt.asset_sweep_summary(result, (("fast", 20), ("slow", 60)))
        s_pa = bt.asset_sweep_summary(result, (("fast", 20), ("slow", 60)),
                                      per_asset_null=True)
        self.assertEqual(s_global.verdict, s_pa.verdict)
        self.assertEqual(s_global.verdict, bt.ASSET_VERDICT_NO_EDGE)

    def test_per_asset_null_one_edge_concentrated(self):
        """A single significant edge must yield CONCENTRATED with one flag."""
        assets = [bt.generate_bars(400, seed=i) for i in range(3)]
        fold_rets = [
            [[0.01, -0.01], [0.00, -0.02]],   # asset 0: near zero
            [[0.00, -0.01], [0.00, -0.03]],   # asset 1: near zero
            [[0.04, 0.05], [0.01, 0.02]],     # asset 2: positive edge
        ]
        result = bt.AssetSweepResult(
            assets=assets,
            asset_names=["a", "b", "c"],
            param_sets=[("fast", 20), ("fast", 40)],
            fold_total_returns=fold_rets,
            baseline_median_log_returns=[-0.02, 0.03, 0.10],
            n_folds=2, n_periods=400,
            train_window_bars=120, test_window_bars=40,
        )
        summary = bt.asset_sweep_summary(result, (("fast", 20),),
                                        per_asset_null=True)
        self.assertIsNotNone(summary.asset_null_medians)
        self.assertIsNotNone(summary.asset_significance)
        self.assertEqual(len(summary.asset_null_medians), 3)
        self.assertEqual(len(summary.asset_significance), 3)
        self.assertEqual(summary.verdict, bt.ASSET_VERDICT_CONCENTRATED)
        self.assertGreater(summary.n_significant_assets, 0)

    def test_per_asset_null_self_consistent(self):
        """Significance flags must match the per-asset tolerance rule."""
        assets = bt.flat_regime_assets(4, seed=99, n_bars=800)
        grid = [{"fast": 20}]
        result = bt.sweep_across_assets(
            lambda closes, **p: [bt.Signal(date=i + 1, weight=0.0)
                                 for i in range(len(closes))],
            assets, grid, (("fast", 20),),
            train_window=120, test_window=40, warmup=10,
        )
        summary = bt.asset_sweep_summary(result, (("fast", 20),),
                                        per_asset_null=True)
        tol = max(0.05, 2.0 * summary.null_dispersion)
        for m, n, sig in zip(
            summary.median_log_returns, summary.asset_null_medians,
            summary.asset_significance,
        ):
            self.assertEqual(sig, abs(m - n) > tol)
        self.assertEqual(summary.n_significant_assets,
                         sum(summary.asset_significance))
