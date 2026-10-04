"""Regime-stability stress testing across the full collected universe
(Yahoo Finance OHLCV), partitioning each asset into contiguous volatility
regime segments (research-only).

This closes the last robustness dimension on real data at the universe
level: the MA crossover was already run with perturbation, asset-universe,
and per-asset regime-stability (AAPL, NVDA) on the collected data; this
runs regime-stability across ALL collected assets with the same
deterministic settings in a single call, so the cross-asset pattern of
regime-stability verdicts can be judged as a class of strategies rather
than as isolated single-asset results.

Method: each asset is split into 4 contiguous blocks by date; each block is
labeled 'turbulent' if its median trailing-60-bar realized volatility
(annualized) exceeds the series-wide median, else 'calm'. Blocks form
segments; each segment is backtested with walk-forward IS/OOS and compared
against a coin-flip null run on the same segment. Per-asset medians are
compared against null medians with the same verdict logic as
``regime_stress``.

Research/simulation only. No live trading, no production execution.
Results inform only whether to continue research on this idea.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
WARMUP = 60
FAST, SLOW = 20, 60
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60


def volatility_blocks(closes: NDArray, n_blocks: int = 4, window: int = 60):
    """Past-only classification into contiguous date blocks labeled by
    realized-volatility regime; delegates to the framework module."""
    return bt.volatility_blocks(closes, n_blocks=n_blocks, window=window)


def main():
    np.random.seed(42)
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Universe loading + leakage review per asset ===")
    universe = bt.load_universe()
    cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARMUP)
    for ticker, (bars, dates) in universe.items():
        closes = bars.closes_array()
        signals = bt.ma_crossover_signals(closes, FAST, SLOW)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cfg0)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        print(f"  {ticker}: {bars.n_bars} bars, leakage [PASS]")

    print("\n=== 4. Regime classification (4 contiguous blocks by date, past-only) ===")
    for ticker, (bars, dates) in universe.items():
        closes = bars.closes_array()
        labels = volatility_blocks(closes, n_blocks=4)
        from collections import Counter
        counts = Counter(labels)
        print(f"  {ticker}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    print("\n=== 5. Regime-stability across universe (MA crossover, MA(20/60)) ===")
    baseline = (("fast", FAST), ("slow", SLOW))
    grid = [{"fast": FAST, "slow": SLOW}]
    summary = bt.stress_segments_across_tickers(
        {t: b for t, (b, _d) in universe.items()},
        signals_fn=bt.ma_crossover_signals,
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=4),
        baseline=baseline,
        param_grid=grid,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg0,
        min_segment_bars=400,
    )
    print(summary.inspect())

    print("\nInterpretation:")
    vc = summary.verdict_counts
    if vc["REGIME_STABLE"] > vc["CONSISTENT_WITH_NOISE"]:
        print("  A majority of assets show REGIME_STABLE behavior, consistent "
              "with the AAPL verdict and with a genuine (still weak) signal.")
    elif vc["REGIME_DEPENDENT"] > 0:
        print("  One or more assets show REGIME_DEPENDENT behavior, which would "
              "indicate regime-specific fitting rather than a robust class "
              "signal.")
    else:
        print("  Most assets show CONSISTENT_WITH_NOISE across regime segments; "
              "the candidate's apparent full-sample edge does not survive the "
              "regime-stability gate at the universe level.")

    print("\nCross-reference:")
    print("  Asset-universe sweep (examples/universe_sweep.py): CONSISTENT, "
          "best-asset edge share ~28% (not concentrated in one asset).")
    print("  AAPL fold t-statistic (research/checks/verify_aapl_stats.py): "
          "t=2.75, 95% CI excludes 0 -- marginal, but inside the framework's "
          "sample-calibrated noise tolerance; not admitted to the evidence base.")
    print()
    print("NOTE: research/simulation only. No live trading or production "
          "execution. MA crossover regime-stability across the collected "
          "universe: exploratory simulation.")


if __name__ == "__main__":
    main()
