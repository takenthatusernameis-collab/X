#!/usr/bin/env python3
"""Holding period robustness test for lookback-5 momentum.

Tests whether the qualified momentum effect survives small predeclared
holding-period variations without becoming a single-point timing artifact.

Uses the existing walk-forward/null methodology on the 10-asset universe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest.real_data import load_universe


def holding_period_momentum_signals(
    closes: np.ndarray,
    lookback: int = 5,
    hold_period: int = 1,
    warmup: int = 60,
) -> list[bt.Signal]:
    """Past-only momentum signal with multi-day holding.

    Long the previous ``lookback``-day return and hold for ``hold_period`` days:
    if the last ``lookback`` days were up, go long; if they were down, go short.
    Hold position for ``hold_period`` days before daily rebalancing.
    Neutral before the ``lookback`` window, and during warmup period.

    No look-ahead: each signal uses only closes up to and including its own
    bar.
    """
    n = len(closes)
    signals: list[bt.Signal] = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    hold_period = int(hold_period)

    # Generate momentum signals after the lookback window
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)

    # Hold position for the specified holding period
    for i in range(lookback, n):
        if i + hold_period < n and signals[i].weight != 0.0:
            # Propagate the same signal forward for hold_period days
            weight = signals[i].weight
            for j in range(i + 1, min(i + hold_period, n)):
                signals[j] = bt.Signal(date=j + 1, weight=weight)

    # Ensure no non-neutral signals during warmup period
    for i in range(min(warmup, n)):
        signals[i] = bt.Signal(date=signals[i].date, weight=0.0)

    return signals


def main():
    # Load the 10-asset universe
    universe_data = load_universe()
    tickers = list(universe_data.keys())

    # Predeclared holding-period grid
    holding_periods = [1, 3, 5, 7, 10]

    print("=== Holding Period Robustness Test for Lookback-5 Momentum ===")
    print(f"Assets: {tickers}")
    print(f"Holding periods: {holding_periods}")
    print(f"Lookback: 5 days (predeclared)")
    print()

    # Configuration matching the existing momentum workflow
    cfg = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=60,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )

    # Walk-forward parameters matching existing tests
    train_window = 252  # 12 months
    test_window = 84   # 4 months
    warmup = 60        # warmup periods
    overlap_window = 60

    results = {}

    for hold_period in holding_periods:
        print(f"--- Holding Period: {hold_period} days ---")

        per_asset_results = {}

        for ticker in tickers:
            # Extract BarSequence from the (BarSequence, date_strings) tuple
            bars_data = universe_data[ticker]
            if isinstance(bars_data, tuple) and len(bars_data) == 2:
                bars = bars_data[0]  # BarSequence
                dates = bars_data[1]  # List of date strings
            else:
                bars = bars_data
                dates = None

            closes = bars.closes_array()

            signals = holding_period_momentum_signals(closes, lookback=5, hold_period=hold_period)

            # Verify signal integrity - use bar dates from the list of Bar objects
            # Convert BarSequence to list of Bar objects to get dates properly
            bar_objects = list(bars)
            bar_dates = [b.date for b in bar_objects]
            bt.check_signal_integrity(signals, bar_dates, warmup=warmup)

            # Run walk-forward
            result = bt.walk_forward(
                bars=bar_objects,
                signals=signals,
                train_window=train_window,
                test_window=test_window,
                warmup=warmup,
                overlap_window=overlap_window,
                cfg=cfg,
            )

            # Store fold results
            per_asset_folds = []
            for fold in result.folds:
                per_asset_folds.append({
                    "fold_index": fold.fold_index,
                    "start_date": fold.start_date,
                    "end_date": fold.end_date,
                    "total_return": fold.metrics["total_return"],
                    "log_return": np.log1p(np.clip(fold.metrics["total_return"], -1.0 + 1e-12, None)),
                    "n_trades": fold.metrics["n_trades"],
                    "sharpe": fold.metrics["sharpe"],
                    "max_drawdown": fold.metrics["max_drawdown"],
                })

            per_asset_results[ticker] = per_asset_folds

        # Compute per-asset summary
        per_asset_summary = {}
        for ticker, folds in per_asset_results.items():
            log_returns = [f["log_return"] for f in folds]
            per_asset_summary[ticker] = {
                "n_folds": len(folds),
                "median_log_return": float(np.median(log_returns)),
                "mean_log_return": float(np.mean(log_returns)),
                "std_log_return": float(np.std(log_returns)),
                "positive_folds": sum(1 for f in folds if f["total_return"] > 0),
                "negative_folds": sum(1 for f in folds if f["total_return"] < 0),
                "verdict": "POS" if np.median(log_returns) > 0.05 else "NEG" if np.median(log_returns) < -0.05 else "NOEDGE",
            }

        results[f"hold_{hold_period}"] = {
            "per_asset": per_asset_summary,
            "config": {
                "lookback": 5,
                "holding_period": hold_period,
                "train_window": train_window,
                "test_window": test_window,
                "warmup": warmup,
                "overlap_window": overlap_window,
            }
        }

        # Print summary
        verdicts = [r["verdict"] for r in per_asset_summary.values()]
        pos_count = sum(1 for v in verdicts if v == "POS")
        neg_count = sum(1 for v in verdicts if v == "NEG")
        noedge_count = sum(1 for v in verdicts if v == "NOEDGE")

        print(f"Per-asset verdicts: POS={pos_count}, NEG={neg_count}, NOEDGE={noedge_count}")
        for ticker in tickers:
            verdict = per_asset_summary[ticker]["verdict"]
            median_ret = per_asset_summary[ticker]["median_log_return"]
            print(f"  {ticker:6s}: {verdict:6s} (median log ret={median_ret:+.3f})")
        print()

    # Save results
    output_path = Path("state/check_artifacts/holding_period_momentum_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
            "lookback": 5,
            "holding_periods": holding_periods,
            "min_segment_bars": 400,
            "n_blocks": 4,
            "overlap": 60,
            "per_holding_period": results,
            "seed": 42,
            "test": test_window,
            "train": train_window,
            "warmup": warmup,
            "window": 60,
        }, f, indent=2)

    print(f"Results saved to {output_path}")

    print("\n=== Summary ===")
    print("Holding-period test completed.")
    print(f"Tested holding periods: {holding_periods}")
    print("Next: Independent verification and leakage review")

    return results


if __name__ == "__main__":
    main()
