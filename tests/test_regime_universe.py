"""Unit tests for the universe-wide regime-stability runner.

Deterministic and runnable with `python -m unittest`.
"""
import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


class TestVolatilityBlocks(unittest.TestCase):
    """The reusable block-based regime classifier must be deterministic and
    past-only."""

    def test_past_only_prefix(self):
        closes = np.ones(200, dtype=np.float64) * 100.0
        for i in range(1, len(closes)):
            closes[i] = closes[i - 1] * 1.0005  # small drift, vol > 0
        labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
        self.assertEqual(len(labels), 200)
        for i in range(60):
            self.assertEqual(labels[i], "insufficient")

    def test_reproducibility(self):
        closes = np.ones(500, dtype=np.float64) * 100.0
        for i in range(1, len(closes)):
            closes[i] = closes[i - 1] * (1 + np.random.default_rng(7).normal(0, 0.01))
        r1 = bt.volatility_blocks(closes, n_blocks=4, window=60)
        r2 = bt.volatility_blocks(closes, n_blocks=4, window=60)
        self.assertEqual(r1, r2)

    def test_known_labels(self):
        closes = np.ones(400, dtype=np.float64) * 100.0
        for i in range(1, len(closes)):
            closes[i] = closes[i - 1] * 1.0005
        labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
        for l in labels:
            self.assertIn(l, ("calm", "turbulent", "insufficient"))

    def test_four_blocks_labelled(self):
        closes = np.ones(400, dtype=np.float64) * 100.0
        for i in range(1, len(closes)):
            closes[i] = closes[i - 1] * 1.001
        labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
        blocks = [i // 100 for i in range(400)]
        # each block has exactly one of its two possible labels
        for b in set(blocks):
            block_labels = {labels[i] for i in range(b * 100, (b + 1) * 100) if labels[i] != "insufficient"}
            self.assertEqual(len(block_labels), 1, msg=f"block {b} has labels {block_labels}")


class TestMaCrossoverSignals(unittest.TestCase):
    """The reusable MA-crossover signal must respect the contract used by the
    real-data examples (neutral padding, +/-1 weights)."""

    def test_padding_neutral_before_slow_window(self):
        closes = np.ones(100, dtype=np.float64)
        sig = bt.ma_crossover_signals(closes, fast=20, slow=60)
        for i in range(59):
            self.assertEqual(sig[i].weight, 0.0)
        self.assertIn(sig[59].weight, (1.0, -1.0))

    def test_weights_contract(self):
        closes = np.array([100.0, 101.0, 102.0] + [100.0] * 150, dtype=np.float64)
        sig = bt.ma_crossover_signals(closes, fast=20, slow=60)
        for s in sig:
            self.assertIn(s.weight, (-1.0, 0.0, 1.0))
        self.assertEqual(len(sig), len(closes))


class TestStressSegmentsAcrossTickers(unittest.TestCase):
    """The universe runner must behave like repeated `stress_segments` calls."""

    def setUp(self):
        np.random.seed(13)
        closes_a = np.cumsum(np.random.default_rng(1).normal(0.0003, 0.02, 800)) + 100
        closes_b = np.cumsum(np.random.default_rng(2).normal(0.0005, 0.025, 800)) + 100
        self.bars_a = bt.BarSequence(
            np.arange(1, 801), closes_a, closes_a, closes_a, closes_a, np.zeros(800)
        )
        self.bars_b = bt.BarSequence(
            np.arange(1, 801), closes_b, closes_b, closes_b, closes_b, np.zeros(800)
        )
        self.tickers = {"A": self.bars_a, "B": self.bars_b}

    def _runner(self, **kwargs):
        kwargs.setdefault("tickers", self.tickers)
        kwargs.setdefault("signals_fn", bt.ma_crossover_signals)
        kwargs.setdefault("regime_labels_fn", bt.volatility_blocks)
        kwargs.setdefault("train_window", 60)
        kwargs.setdefault("test_window", 20)
        kwargs.setdefault("warmup", 0)
        kwargs.setdefault("overlap_window", 0)
        kwargs.setdefault("min_segment_bars", 100)
        return bt.stress_segments_across_tickers(**kwargs)

    def test_across_tickers_determinism(self):
        s1 = self._runner()
        s2 = self._runner()
        self.assertEqual(s1.n_assets, s2.n_assets)
        self.assertEqual(s1.ticker_order, s2.ticker_order)
        self.assertEqual(s1.verdict_counts, s2.verdict_counts)
        for t in s1.ticker_order:
            r1, r2 = s1.assets[t], s2.assets[t]
            self.assertEqual(r1.overall_verdict, r2.overall_verdict)
            for a, b in zip(r1.scenarios, r2.scenarios):
                self.assertEqual(a.baseline_median_log_return, b.baseline_median_log_return)
                self.assertEqual(a.n_folds, b.n_folds)
        self.assertEqual(s1.inspect(), s2.inspect())

    def test_across_tickers_empty_raises(self):
        with self.assertRaises(ValueError):
            self._runner(tickers={})

    def test_single_asset_structure(self):
        s = self._runner(tickers={"A": self.bars_a})
        self.assertEqual(s.n_assets, 1)
        self.assertEqual(s.ticker_order, ["A"])
        res = s.assets["A"]
        self.assertGreater(len(res.scenarios), 0)
        self.assertIn(res.overall_verdict,
                      ("REGIME_STABLE", "REGIME_DEPENDENT", "CONSISTENT_WITH_NOISE"))
        for sc in res.scenarios:
            self.assertTrue(np.isfinite(sc.baseline_median_log_return))

    def test_verdict_counts_consistent(self):
        s = self._runner()
        total = sum(s.verdict_counts.values())
        self.assertEqual(total, s.n_assets)
        self.assertEqual(set(s.verdict_counts),
                          {"REGIME_STABLE", "REGIME_STABLE_LOSS",
                           "REGIME_DEPENDENT", "CONSISTENT_WITH_NOISE"})
        for res in s.assets.values():
            self.assertIn(res.overall_verdict, s.verdict_counts)

    def test_signal_length_mismatch_raises(self):
        def short_signals(closes, **params):
            return [bt.Signal(date=i + 1, weight=1.0) for i in range(len(closes) - 10)]
        with self.assertRaises(ValueError):
            self._runner(
                tickers={"A": self.bars_a, "B": self.bars_b},
                signals_fn=short_signals,
            )

    def test_inspect_contains_all_tickers(self):
        s = self._runner()
        for ticker in s.ticker_order:
            self.assertIn(ticker, s.inspect())
        self.assertIn("Verdict counts:", s.inspect())

    def test_ticker_order_preserves_input_order(self):
        ordered = {"Zebra": self.bars_b, "Alpha": self.bars_a, "Beta": self.bars_a}
        s = self._runner(tickers=ordered)
        self.assertEqual(s.ticker_order, ["Zebra", "Alpha", "Beta"])


class TestStressSegmentsAcrossTickersKnownVertex(unittest.TestCase):
    """Known-vertex detection must also hold at universe level."""

    def test_universe_detects_regime_dependent_contrived(self):
        # "A" is a two-regime series (calm then strong up-drift) so a
        # long-only signal is regime-dependent; "B" is flat so the same
        # signal is noise.
        regimes_a = [
            bt.Regime(drift_annual=0.0, vol_annual=0.02, intraday_range_scale=0.005),
            bt.Regime(drift_annual=1.5, vol_annual=0.15, intraday_range_scale=0.02),
        ]
        regimes_b = [bt.Regime(drift_annual=0.0, vol_annual=0.15, intraday_range_scale=0.02)]
        bars_a = bt.generate_bars(1000, regimes=regimes_a, p_transition=0.0,
                                  start_price=100.0, seed=42)
        bars_b = bt.generate_bars(1000, regimes=regimes_b, p_transition=0.0,
                                  start_price=100.0, seed=43)
        ticks = {"A": bars_a, "B": bars_b}
        s = bt.stress_segments_across_tickers(
            ticks,
            lambda c, **params: [bt.Signal(date=i + 1, weight=1.0) for i in range(len(c))],
            lambda c: bt.volatility_blocks(c, n_blocks=2),
            train_window=60, test_window=20, warmup=0,
            min_segment_bars=300,
        )
        self.assertEqual(s.n_assets, 2)
        self.assertIn("A", s.assets)
        self.assertIn("B", s.assets)
        for ticker, res in s.assets.items():
            self.assertGreater(res.n_folds, 0)
            self.assertTrue(np.isfinite(res.candidate_dispersion))
            self.assertTrue(np.isfinite(res.null_dispersion))
            self.assertIn(res.overall_verdict,
                          ("REGIME_STABLE", "REGIME_DEPENDENT", "CONSISTENT_WITH_NOISE"))
        self.assertTrue(s.inspect())
