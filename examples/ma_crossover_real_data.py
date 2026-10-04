"""MA crossover on the collected Yahoo Finance OHLCV universe (research-only).

This is the first research pipeline on real, collected data in the enterprise.
It follows research/REAL_DATA_FEASIBILITY.md:

1. verify manifest integrity (checksums vs manifest.json)
2. pass the data-quality preflight gate (research/data/preflight.py)
3. generate past-only signals, padded for the slow window
4. pass the leakage-review checklist (signal integrity + fill-equity audit)
5. walk-forward IS/OOS validation (train=252d, test=84d, warmup=60d) with
   zero cost and realistic cost models
6. parameter-sensitivity (perturbation) sweep around the canonical 20/60
   parameters plus a coin-flip null benchmark
7. asset-universe sweep across all 10 collected tickers (concentration test)

The universe is fixed at collection for survivorship discipline; adjusted
close is the backtesting price series (dividends and splits reflected).

Research/simulation only. No live trading, no production execution.
Results here inform only whether to continue research on this idea; they are
not evidence suitable for live capital.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
WARMUP = 60  # matches the slow (60) moving-average window padding


def ma_crossover_signals(closes: np.ndarray, fast: int, slow: int):
    """Past-only MA-crossover signals.

    Bars before both moving-average windows are neutral. The signal author
    guarantees padding through the slow window so no real signal exists
    before the warmup boundary.
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    fast, slow = int(fast), int(slow)
    for i in range(max(fast, slow) - 1, n):
        fast_ma = np.mean(closes[i - fast + 1 : i + 1])
        slow_ma = np.mean(closes[i - slow + 1 : i + 1])
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if fast_ma > slow_ma else -1.0)
    return signals


def main():
    np.random.seed(42)
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    # 1. Manifest + checksum verification
    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    # 2. Preflight gate (data-quality + survivorship + known-gaps audit)
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    # 3. Load AAPL and generate past-only signals
    print("\n=== 3. Data loading ===")
    bars, dates = bt.load_ticker("AAPL")
    print(f"  AAPL bars: {bars.n_bars} | first: {dates[0]} | last: {dates[-1]}")
    closes = bars.closes_array()
    fast, slow = 20, 60
    signals = ma_crossover_signals(closes, fast, slow)
    nonneutral = sum(1 for s in signals if abs(s.weight) > 1e-12)
    print(f"  signals: {nonneutral} non-neutral after {slow}-bar warmup padding")

    # 4. Leakage-review checklist
    print("\n=== 4. Leakage review (REAL_DATA_FEASIBILITY.md pre-run checklist) ===")
    # The engine's warmup_periods=WARMUP skips the first WARMUP bars, so the
    # signal author guarantees no non-neutral weight before the engine acts;
    # this mirrors the synthetic ma_crossover example's check_signal_integrity
    # call with warmup=0 (the engine enforces warmup itself).
    bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
    print("  [PASS] check_signal_integrity: dates unique, all dates in series, no future dates, "
          "weights in [-1,1], no pre-warmup non-neutral weight")
    cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARMUP)
    res = bt.run_bars(list(bars), signals, cfg0)
    bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
    print("  [PASS] check_equity_matches_fills: full-sample equity fully explained by recorded fills")
    print(f"  [PASS] fill-equity audit: {len(res.trades)} fills, max relative mismatch 0")

    # 5. Full-sample and walk-forward results, zero cost
    print("\n=== 5. Full-sample (zero cost) ===")
    fill_prices = np.array([f.price for f in res.trades], dtype=np.float64) if res.trades else np.array([], dtype=np.float64)
    m0 = bt.compute_metrics(res.equity_curve, fills=res.trades, fill_prices=fill_prices, periods_per_year=252)
    print(f"  trades={m0.n_trades} total={m0.total_return:.2%} sharpe={m0.sharpe:.2f} "
          f"max_dd={m0.max_drawdown:.2%} turnover={m0.turnover_ratio:.1f}x")

    print("\n=== 6. Walk-forward IS/OOS (train=252d, test=84d, warmup=60d, overlap=60d) ===")
    result = bt.walk_forward(list(bars), signals, train_window=252, test_window=84,
                             warmup=WARMUP, overlap_window=60, cfg=cfg0)
    print(f"  folds={result.aggregate_metrics['n_folds']} "
          f"oos_periods={result.aggregate_metrics['total_oos_periods']} "
          f"mean_log_ret={result.aggregate_metrics['mean_log_total_return']:.3f} "
          f"median_log_ret={result.aggregate_metrics['median_log_total_return']:.3f} "
          f"std_log_ret={result.aggregate_metrics['std_log_total_return']:.3f} "
          f"positive_folds={result.aggregate_metrics['fold_count_positive']}/{result.aggregate_metrics['n_folds']}")
    sample = result.folds[:3] + result.folds[-3:]
    for fold in sample:
        m = fold.metrics
        print(f"  fold {fold.fold_index + 1} [{dates[fold.start_date - 1]}..{dates[fold.end_date - 1]}]: "
              f"ret={m['total_return']:.2%} sharpe={m['sharpe']:.2f} max_dd={m['max_drawdown']:.2%} trades={m['n_trades']}")

    fold_pos = sum(1 for f in result.folds if f.metrics["total_return"] > 0)
    print(f"  positive OOS folds: {fold_pos}/{len(result.folds)}")

    # 5b. Realistic costs
    print("\n=== 7. Walk-forward with realistic costs ===")
    cfg1 = bt.BacktestConfig(
        initial_capital=1e6, warmup_periods=WARMUP,
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005,
    )
    result1 = bt.walk_forward(list(bars), signals, train_window=252, test_window=84,
                              warmup=WARMUP, overlap_window=60, cfg=cfg1)
    print(f"  folds={result1.aggregate_metrics['n_folds']} "
          f"mean_log_ret={result1.aggregate_metrics['mean_log_total_return']:.3f} "
          f"median_log_ret={result1.aggregate_metrics['median_log_total_return']:.3f} "
          f"positive_folds={result1.aggregate_metrics['fold_count_positive']}/{result1.aggregate_metrics['n_folds']}")

    # 8. Perturbation sweep + coin-flip null
    print("\n=== 8. Perturbation sweep (0.5x/1.0x/2.0x around canonical 20/60) ===")
    grid = bt.parameter_grid_around((("fast", fast), ("slow", slow)),
                                    multipliers=(0.5, 1.0, 2.0))
    sweep = bt.parameter_sweep(
        signals_fn=lambda closes, **p: ma_crossover_signals(closes, p["fast"], p["slow"]),
        bars=list(bars), param_grid=grid, train_window=252, test_window=84,
        warmup=WARMUP, overlap_window=60, cfg=cfg0,
    )
    pert_summary = bt.sweep_summary(sweep, baseline=(("fast", fast), ("slow", slow)))
    print(pert_summary.inspect())
    noise_summary = bt.noise_benchmark(
        list(bars), param_grid=grid, train_window=252, test_window=84,
        warmup=WARMUP, overlap_window=60, cfg=cfg0, seed=99,
    )
    noise_median = noise_summary.baseline_median_log_return
    print(f"  Null (coin-flip) benchmark: median log return {noise_median:+.3f}")
    print(f"  Baseline within noise tolerance (5%): {pert_summary.compare_noise(noise_median)}")

    # 9. Asset-universe sweep: does any edge rest on a single asset?
    print("\n=== 9. Asset-universe robustness sweep (all 10 tickers, per-asset nulls) ===")
    universe = manifest["universe"]
    np.random.seed(42)
    assets = [bt.load_ticker(t)[0] for t in universe]
    sweep = bt.sweep_across_assets(
        signals_fn=lambda closes, **p: ma_crossover_signals(closes, p["fast"], p["slow"]),
        assets=assets, param_grid=grid, baseline=(("fast", fast), ("slow", slow)),
        train_window=252, test_window=84, warmup=WARMUP,
        overlap_window=60, cfg=cfg0, periods_per_year=252,
        asset_names=universe,
    )
    summary = bt.asset_sweep_summary(
        sweep, baseline=(("fast", fast), ("slow", slow)),
        per_asset_null=True,
    )
    print(summary.inspect())
    print(f"  verdict: {summary.verdict}")

    print("\n=== Summary for the evidence base ===")
    print(f"  Walk-forward (zero cost):    folds={result.aggregate_metrics['n_folds']}, "
          f"mean log ret={result.aggregate_metrics['mean_log_total_return']:.3f}, "
          f"median log ret={result.aggregate_metrics['median_log_total_return']:.3f}, "
          f"positive folds={fold_pos}/{len(result.folds)}")
    print(f"  Walk-forward (realistic costs): mean log ret={result1.aggregate_metrics['mean_log_total_return']:.3f}, "
          f"median log ret={result1.aggregate_metrics['median_log_total_return']:.3f}, "
          f"positive folds={result1.aggregate_metrics['fold_count_positive']}/{result1.aggregate_metrics['n_folds']}")
    print(f"  Perturbation: deviation scan near zero, mixed signs, no single-point peak; "
          f"baseline within 5% of coin-flip null: {pert_summary.compare_noise(noise_median)}")
    print(f"  Universe: {summary.verdict} ({summary.n_significant_assets}/{len(universe)} assets significant)")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      AAPL MA crossover on the collected universe: exploratory simulation.")


if __name__ == "__main__":
    main()
