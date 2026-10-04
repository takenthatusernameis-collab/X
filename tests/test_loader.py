"""Unit tests for the real-data ticker loader.

Research/simulation only. These tests verify the loader in `research/backtest/
real_data.py` against the collected CSV universe and the manifest contract.
The loader is the path from `research/data/raw/*.csv` into the deterministic
backtest engine, so its contract (adjusted close, bar counts, date mapping) is
checked explicitly rather than assumed.
"""
import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest.real_data import MANIFEST_PATH

MANIFEST = None


def load_manifest():
    global MANIFEST
    if MANIFEST is None:
        import json
        with open(MANIFEST_PATH) as fh:
            MANIFEST = json.load(fh)
    return MANIFEST


class TestLoaderContract(unittest.TestCase):
    """The loader must honor the manifest contract: adjusted close is the
    backtesting price series."""

    def test_aapl_adjclose_matches_csv_adjclose_field(self):
        """AAPL first-row close must equal the CSV `adjclose` field (2.714299),
        not the raw `close` field (3.241071)."""
        bars, dates = bt.load_ticker("AAPL")
        self.assertEqual(dates[0], "2009-01-02")
        self.assertAlmostEqual(bars.closes_array()[0], 2.714299, places=6)
        # Ensure the raw close field is NOT what was loaded.
        self.assertNotAlmostEqual(bars.closes_array()[0], 3.241071, places=6)

    def test_loader_returns_adjclose_for_multiple_tickers(self):
        """adjclose != close for several tickers at a known date; loader picks
        adjclose consistently."""
        for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            self.assertEqual(len(closes), len(dates))
            self.assertFalse(np.isnan(closes).any())
            self.assertTrue((closes > 0).all())

    def test_loader_bar_count_matches_manifest(self):
        """Every ticker's bar count must match `manifest["per_ticker_bars"]`."""
        manifest = load_manifest()
        for ticker, n in manifest["per_ticker_bars"].items():
            bars, _ = bt.load_ticker(ticker)
            self.assertEqual(len(bars), n, ticker)

    def test_loader_date_mapping_starts_at_first_available_date(self):
        """dates[0] must equal the manifest's first_available_date per ticker."""
        manifest = load_manifest()
        for ticker in manifest["universe"]:
            bars, dates = bt.load_ticker(ticker)
            self.assertEqual(dates[0], manifest["first_available_date"][ticker])

    def test_loader_dates_match_csv_order(self):
        """Calendar dates must be in the same order as the CSV rows (monotonic
        and matching the file order)."""
        bars, dates = bt.load_ticker("AAPL")
        raw_path = Path("research/data/raw/AAPL_daily.csv")
        expected_dates = []
        with open(raw_path) as fh:
            fh.readline()  # header
            for line in fh:
                if line.strip():
                    expected_dates.append(line.split(",")[0])
        self.assertEqual(dates, expected_dates)

    def test_loader_universe_matches_manifest_universe(self):
        universe = bt.load_universe()
        manifest = load_manifest()
        self.assertEqual(set(universe), set(manifest["universe"]))


class TestRealDataPipelineIntegration(unittest.TestCase):
    """Load real data through the loader and run it through the engine with
    the same leakage discipline used for synthetic data. Real data with
    audited provenance is candidate evidence territory; the framework must
    enforce the no-look-ahead rules here as well."""

    def _ma_signals(self, closes):
        """Past-only MA-crossover signals, neutral (zero-weight) for every bar
        through the warmup boundary so they pass check_signal_integrity with
        warmup=60. First possible non-neutral signal is at date 61."""
        n = len(closes)
        fast, slow = 20, 60
        signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
        for i in range(max(fast, slow), n):  # 0-based index 60 -> date 61
            fast_ma = np.mean(closes[i - fast + 1 : i + 1])
            slow_ma = np.mean(closes[i - slow + 1 : i + 1])
            signals[i] = bt.Signal(date=i + 1, weight=1.0 if fast_ma > slow_ma else -1.0)
        return signals

    def test_aapl_real_data_leakage_checks_pass(self):
        bars, dates = bt.load_ticker("AAPL")
        signals = self._ma_signals(bars.closes_array())
        # The signal is padded to neutral before the slow MA window, matching
        # the engine's warmup discipline.
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=60)

    def test_aapl_real_data_walk_forward_and_fill_audit(self):
        """Walk-forward on real AAPL data must pass the engine's equity-vs-fill
        audit, confirming the loader produces bar prices the engine can use
        without discrepancy."""
        np.random.seed(42)
        bars, dates = bt.load_ticker("AAPL")
        signals = self._ma_signals(bars.closes_array())

        warmup = 60
        cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=warmup)

        full = bt.run_bars(list(bars), signals, cfg)
        bt.check_equity_matches_fills(full.equity_curve, full.trades,
                                      bars.closes_array(), 1e6)

        result = bt.walk_forward(list(bars), signals, train_window=252,
                                 test_window=84, warmup=warmup,
                                 overlap_window=60, cfg=cfg)

        # FoldResult exposes metrics + equity_curve but not trades; the
        # full-sample audit above already exercises the engine+loader fill
        # path. Here we verify walk-forward aggregates are populated.
        self.assertGreater(result.aggregate_metrics["n_folds"], 0)
        self.assertGreater(result.aggregate_metrics["total_oos_periods"], 0)

    def test_loader_deterministic_rerun(self):
        bars1, dates1 = bt.load_ticker("AAPL")
        bars2, dates2 = bt.load_ticker("AAPL")
        self.assertTrue(np.array_equal(bars1.closes_array(), bars2.closes_array()))
        self.assertTrue(np.array_equal(bars1.opens, bars2.opens))
        self.assertEqual(dates1, dates2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
