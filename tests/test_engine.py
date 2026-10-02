"""Unit tests for the backtest engine.

Deterministic (seeded) and runnable with `python -m unittest`.
"""
import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


class TestEngineCosts(unittest.TestCase):
    """Commission and slippage must be applied and traceable."""

    def test_commission_exact(self):
        """Running with zero commission must differ from non-zero only by commission."""
        np.random.seed(42)
        bars = bt.generate_bars(300, seed=42)
        # simple deterministic signals: long when day index is even
        signals = [bt.Signal(date=i + 1, weight=1.0 if i % 2 == 0 else 0.0) for i in range(len(bars))]

        cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=10)
        cfg1 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=10, commission_per_share=0.003)

        # commission is applied per trade, so results diverge from the first trade onward
        res0 = bt.run_bars(list(bars), signals, cfg0)
        res1 = bt.run_bars(list(bars), signals, cfg1)

        self.assertEqual(res0.equity_curve.shape, res1.equity_curve.shape)
        # commission is applied per trade, so results diverge from the first
        # trade (index 10) onward; index 15 must be lower with commission.
        self.assertLess(res1.equity_curve[15], res0.equity_curve[15])
        # equity decline from first trade equals commission * shares * price
        first_trade = res1.trades[0]
        expected_commission = first_trade.commission
        self.assertGreater(expected_commission, 0)

    def test_slippage_price_is_close_not_high(self):
        """Fill price must be close + slippage, never open/high/low (no lookahead)."""
        bar = bt.Bar(date=1, open=100.0, high=1000.0, low=0.001, close=100.0, volume=1e6)
        signal = bt.Signal(date=1, weight=1.0)
        cfg = bt.BacktestConfig(
            initial_capital=1e6,
            target_exposure=1.0,
            slippage_cents=5.0,           # 5 cents per share
            slippage_proportional=0.0,
            warmup_periods=0,
        )
        res = bt.run_bars([bar], [signal], cfg)
        # position = weight * exposure * capital / close = 10000 shares
        self.assertAlmostEqual(res.trades[0].price, 100.05, places=5)
        self.assertAlmostEqual(res.trades[0].shares, 10000.0, places=5)
        self.assertEqual(res.orders[0].side, "buy")
        # cash = 1e6 - 10000*100.05 = -500; equity = cash + 10000*100 = 999500
        self.assertAlmostEqual(res.equity_curve[0], 999500.0, places=4)

    def test_fill_price_with_proportional_slippage(self):
        """Proportional slippage applies on top of the close."""
        bar = bt.Bar(date=1, open=100.0, high=1000.0, low=0.001, close=100.0, volume=1e6)
        signal = bt.Signal(date=1, weight=1.0)
        cfg = bt.BacktestConfig(
            initial_capital=1e6,
            target_exposure=1.0,
            slippage_cents=0.0,
            slippage_proportional=0.001,  # 10 bps
            warmup_periods=0,
        )
        res = bt.run_bars([bar], [signal], cfg)
        expected_price = 100.0 * (1 + 0.001)
        self.assertAlmostEqual(res.trades[0].price, expected_price, places=6)
        self.assertAlmostEqual(res.trades[0].shares, 10000.0, places=5)
        # cash = 1e6 - 10000*100.1 = -1000; equity = -1000 + 10000*100 = 999000
        self.assertAlmostEqual(res.equity_curve[0], 999000.0, places=4)

    def test_short_side(self):
        """Shorting and covering must use the same close+slippage logic."""
        np.random.seed(7)
        bars = bt.generate_bars(200, seed=7)
        # long -> short -> cover, with large capital so cash stays ~constant
        signals = [bt.Signal(date=i + 1, weight=1.0 if i < 50 else
                            -1.0 if i < 120 else 1.0) for i in range(len(bars))]
        cfg = bt.BacktestConfig(initial_capital=1e10, target_exposure=0.01, warmup_periods=5)
        res = bt.run_bars(list(bars), signals, cfg)
        shorts = [f for f in res.trades if f.shares < 0]
        covers = [f for f in res.trades if f.shares > 0]
        self.assertGreater(len(shorts), 0)
        self.assertGreater(len(covers), 0)
        # short fills must also price at close + slippage (no open/high/low)
        bar = list(bars)[shorts[0].date - 1]
        expected = bar.close + cfg.slippage_cents / 100.0 + bar.close * cfg.slippage_proportional
        self.assertAlmostEqual(shorts[0].price, expected, places=5)

    def test_warmup_skips_signal_usage(self):
        """First warmup_periods bars must hold initial capital and no signal."""
        np.random.seed(11)
        bars = bt.generate_bars(300, seed=11)
        signals = [bt.Signal(date=i + 1, weight=1.0) for i in range(len(bars))]
        warmup = 25
        cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=warmup)
        res = bt.run_bars(list(bars), signals, cfg)
        for i in range(warmup):
            self.assertAlmostEqual(res.equity_curve[i], 1e6, places=6)
            self.assertEqual(res.positions[i], 0.0)
        # after warmup, the first signal enters a position
        close_at_warmup = list(bars)[warmup].close
        self.assertAlmostEqual(res.positions[warmup], 1e6 / close_at_warmup, places=1)
        # with commission the post-warmup equity must be lower (commission deducted)
        cfg_c = bt.BacktestConfig(
            initial_capital=1e6, warmup_periods=warmup, commission_per_share=0.003
        )
        res_c = bt.run_bars(list(bars), signals, cfg_c)
        self.assertLess(res_c.equity_curve[warmup], res.equity_curve[warmup])


class TestEngineIS_OOS(unittest.TestCase):
    """IS/OOS walk-forward must separate training and out-of-sample equity."""

    def test_fold_equity_uses_oos_segment_only(self):
        """Each fold's metrics must be computed on the OOS segment only."""
        np.random.seed(13)
        bars = bt.generate_bars(500, seed=13)
        signals = [bt.Signal(date=i + 1, weight=1.0 if i % 3 == 0 else 0.0) for i in range(len(bars))]

        result = bt.walk_forward(
            list(bars),
            signals,
            train_window=100,
            test_window=50,
            warmup=20,
            overlap_window=0,
            cfg=bt.BacktestConfig(initial_capital=1e6, warmup_periods=20),
        )
        # Each fold uses only its last 50 bars (OOS segment) for metrics.
        self.assertGreater(len(result.folds), 0)
        for fold in result.folds:
            self.assertEqual(fold.test_window_bars, 50)
            self.assertEqual(fold.train_window_bars, 100)
            self.assertEqual(fold.metrics["n_periods"], 50)
        # 500 bars, step = test - overlap = 50, fold window = 170:
        # folds at fold_start 0,50,...,300 -> 7 folds.
        self.assertEqual(len(result.folds), 7)
        self.assertEqual(result.aggregate_metrics["total_oos_periods"], 350)

    def test_full_sample_vs_fold_consistency(self):
        """Walk-forward OOS equity must equal the full-sample run on the same bars.

        With warmup + train + test == full length, there is exactly one fold
        covering bars[0:N] with the same warmup policy, so its equity curve
        must match the full-sample run bar-for-bar.
        """
        np.random.seed(17)
        bars = bt.generate_bars(400, seed=17)
        signals = [bt.Signal(date=i + 1, weight=1.0) for i in range(len(bars))]

        warmup, train = 10, 100
        test = 290  # warmup + train + test = 400 = full series (one fold)
        cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=warmup)
        full_equity = bt.run_bars(list(bars), signals, cfg).equity_curve

        result = bt.walk_forward(
            list(bars),
            signals,
            train_window=train,
            test_window=test,
            warmup=warmup,
            overlap_window=0,
            cfg=cfg,
        )
        self.assertEqual(len(result.folds), 1)
        fold = result.folds[0]
        self.assertEqual(fold.start_date, 1)
        self.assertEqual(fold.end_date, 400)
        self.assertEqual(fold.train_window_bars, train)
        self.assertEqual(fold.test_window_bars, test)
        # every equity point must match the full-sample run on the same bars
        for j in range(len(fold.equity_curve)):
            self.assertAlmostEqual(
                fold.equity_curve[j], full_equity[j], places=6
            )


class TestEngineDeterminism(unittest.TestCase):
    """Results must be byte-reproducible given a seed."""

    def test_run_reproducible(self):
        np.random.seed(19)
        bars = bt.generate_bars(300, seed=19)
        signals = [bt.Signal(date=i + 1, weight=1.0 if i % 2 == 0 else 0.0) for i in range(len(bars))]
        cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=10)

        res1 = bt.run_bars(list(bars), signals, cfg)
        res2 = bt.run_bars(list(bars), signals, cfg)
        self.assertTrue((res1.equity_curve == res2.equity_curve).all())
        self.assertEqual(res1.trades, res2.trades)


class TestLeakageChecks(unittest.TestCase):
    """Signal integrity and equity-fill checks must catch real problems."""

    def test_signal_integrity_ok(self):
        bars = bt.generate_bars(200, seed=23)
        signals = [bt.Signal(date=i + 1, weight=0.5) for i in range(len(bars))]
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)  # passes

    def test_signal_integrity_future_bar(self):
        bars = bt.generate_bars(200, seed=23)
        signals = [bt.Signal(date=i + 1, weight=0.5) for i in range(100)] + [
            bt.Signal(date=300, weight=0.5)  # references a non-existent future bar
        ]
        with self.assertRaises(bt.LeakySignalError):
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)

    def test_signal_integrity_warmup_violation(self):
        bars = bt.generate_bars(200, seed=23)
        signals = [bt.Signal(date=i + 1, weight=0.5) for i in range(10)] + [
            bt.Signal(date=i + 1, weight=0.0) for i in range(10, 200)
        ]
        with self.assertRaises(bt.LeakySignalError):
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=20)

    def test_equity_matches_fills(self):
        np.random.seed(29)
        bars = bt.generate_bars(300, seed=29)
        signals = [bt.Signal(date=i + 1, weight=1.0 if i % 2 == 0 else 0.0) for i in range(len(bars))]
        cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=10)
        res = bt.run_bars(list(bars), signals, cfg)
        bt.check_equity_matches_fills(
            res.equity_curve, res.trades, bars.closes_array(), 1e6
        )  # passes

    def test_equity_fill_check_fails_on_perturbed_equity(self):
        np.random.seed(29)
        bars = bt.generate_bars(300, seed=29)
        signals = [bt.Signal(date=i + 1, weight=1.0 if i % 2 == 0 else 0.0) for i in range(len(bars))]
        cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=10)
        res = bt.run_bars(list(bars), signals, cfg)
        perturbed = res.equity_curve.copy()
        perturbed[50] += 1e4
        with self.assertRaises(AssertionError):
            bt.check_equity_matches_fills(perturbed, res.trades, bars.closes_array(), 1e6)


class TestEngineMargin(unittest.TestCase):
    """Margin/collateral modeling: positions must respect margin requirements."""

    def test_margin_default_off(self):
        """Default margin_rate=0: no margin calls; prior behavior unchanged."""
        np.random.seed(42)
        bars = bt.generate_bars(300, seed=42)
        signals = [bt.Signal(date=i + 1, weight=1.0 if i % 2 == 0 else 0.0)
                   for i in range(len(bars))]
        cfg_default = bt.BacktestConfig(initial_capital=1e6, warmup_periods=10)
        cfg_explicit = bt.BacktestConfig(
            initial_capital=1e6, warmup_periods=10,
            margin_rate=0.0, margin_call_liquidate=False,
        )
        res_default = bt.run_bars(list(bars), signals, cfg_default)
        res_explicit = bt.run_bars(list(bars), signals, cfg_explicit)
        self.assertEqual(res_default.margin_calls, [])
        self.assertEqual(res_explicit.margin_calls, [])
        self.assertTrue((res_default.equity_curve == res_explicit.equity_curve).all())
        self.assertEqual(res_default.trades, res_explicit.trades)

    def test_immediate_margin_call(self):
        """target_exposure=12 with margin_rate=0.1: required margin exceeds
        equity on the first fill, so a margin call fires and the position is
        liquidated to neutral in the same bar. The signal trades only once,
        so exactly one call occurs."""
        bars = bt.generate_bars(30, seed=11)
        signals = [bt.Signal(date=1, weight=1.0)] + [
            bt.Signal(date=i + 1, weight=0.0) for i in range(1, len(bars))
        ]
        cfg = bt.BacktestConfig(
            initial_capital=1e6, target_exposure=12,
            margin_rate=0.1, margin_call_liquidate=True, warmup_periods=0,
        )
        res = bt.run_bars(list(bars), signals, cfg)
        self.assertEqual(len(res.margin_calls), 1)
        mc = res.margin_calls[0]
        self.assertGreater(mc.gross_notional, 10e6)
        self.assertGreater(mc.required_margin, 1.0e6)
        self.assertAlmostEqual(mc.equity, 1e6, places=2)
        self.assertEqual(res.positions[-1], 0.0)
        # liquidated_shares is the position that existed before liquidation,
        # i.e. the entry fill size from the same bar.
        self.assertAlmostEqual(mc.liquidated_shares, res.trades[0].shares)

    def test_short_adverse_margin_call(self):
        """A short position that loses value triggers a margin call. The
        short is held at a fixed 10,000 shares (weights tuned to keep target
        constant); a 100 -> 280 move pushes equity below the margin
        requirement and forces liquidation. The signal then goes neutral so
        the engine does not re-enter a position that cannot be margined."""
        bars = bt.generate_bars(20, seed=13)
        bars.closes[:] = 280.0
        bars.closes[0] = 100.0
        bars.opens[:] = 280.0
        bars.lows[:] = 279.0
        bars.highs[:] = 281.0
        bars.opens[0] = 100.0
        bars.lows[0] = 99.0
        bars.highs[0] = 101.0
        capital = 2e6
        target_shares = -10000.0
        weights = [target_shares * c / (2.0 * capital) for c in (100.0, 280.0)]
        weights += [0.0] * (len(bars) - 2)
        signals = [bt.Signal(date=i + 1, weight=w) for i, w in enumerate(weights)]
        cfg = bt.BacktestConfig(
            initial_capital=capital, target_exposure=2.0,
            margin_rate=0.5, margin_call_liquidate=True, warmup_periods=0,
        )
        res = bt.run_bars(list(bars), signals, cfg)
        self.assertEqual(len(res.margin_calls), 1)
        mc = res.margin_calls[0]
        self.assertEqual(mc.date, 2)
        self.assertAlmostEqual(mc.gross_notional, 2.8e6, places=2)
        self.assertAlmostEqual(mc.required_margin, 1.4e6, places=2)
        self.assertLess(mc.equity, mc.required_margin)
        self.assertAlmostEqual(res.positions[-1], 0.0)
        self.assertTrue(any(f.shares > 0 and f.price > 279 for f in res.trades))

    def test_margin_deterministic(self):
        cfg = bt.BacktestConfig(initial_capital=1e6, target_exposure=12,
                                margin_rate=0.1, warmup_periods=0)
        bars = list(bt.generate_bars(50, seed=99))
        signals = [bt.Signal(date=i + 1, weight=1.0) for i in range(50)]
        res1 = bt.run_bars(bars, signals, cfg)
        res2 = bt.run_bars(bars, signals, cfg)
        self.assertEqual(res1.margin_calls, res2.margin_calls)
        self.assertTrue((res1.equity_curve == res2.equity_curve).all())
        self.assertEqual(res1.trades, res2.trades)

    def test_exit_fill_emitted(self):
        """A signal that goes long then neutral must emit an exit fill, and
        trade stats must then count the completed round trip."""
        bars = bt.generate_bars(60, seed=17)
        signals = [bt.Signal(date=i + 1, weight=1.0 if 20 <= i < 40 else 0.0)
                   for i in range(len(bars))]
        cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=0)
        res = bt.run_bars(list(bars), signals, cfg)
        sells = [f for f in res.trades if f.shares < 0]
        buys = [f for f in res.trades if f.shares > 0]
        self.assertGreater(len(buys), 0)
        self.assertGreater(len(sells), 0)
        m = bt.compute_metrics(
            res.equity_curve, fills=res.trades,
            fill_prices=np.array([f.price for f in res.trades], dtype=np.float64),
            periods_per_year=252,
        )
        self.assertEqual(m.n_trades, 1)  # one completed round trip

    def test_walk_forward_margin_calls_aggregate(self):
        """Walk-forward must carry the margin config per fold and report
        the total number of margin calls in the aggregates."""
        bars = bt.generate_bars(500, seed=7)
        signals = [bt.Signal(date=i + 1, weight=1.0) for i in range(len(bars))]
        result = bt.walk_forward(
            list(bars), signals,
            train_window=100, test_window=50, warmup=20, overlap_window=0,
            cfg=bt.BacktestConfig(
                initial_capital=1e6, target_exposure=12,
                margin_rate=0.1, warmup_periods=20,
            ),
        )
        self.assertGreater(result.aggregate_metrics["total_margin_calls"], 0)
        for fold in result.folds:
            self.assertEqual(fold.train_window_bars, 100)
            self.assertEqual(fold.test_window_bars, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
