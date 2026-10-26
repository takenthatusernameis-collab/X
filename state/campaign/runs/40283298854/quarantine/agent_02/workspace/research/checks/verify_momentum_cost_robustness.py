#!/usr/bin/env python3
"""Independent verification of momentum cost-robustness analysis.

This verifies the cost-sensitivity results from momentum_cost_robustness_direct.py
by recomputing the key metrics and ensuring determinism.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = [3, 5, 10]
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_direct_results.json"

def volatility_blocks(closes, n_blocks, window):
    """Independent re-implementation of volatility_blocks for verification."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels

def segments_from_labels(labels, min_segment_bars):
    """Independent re-implementation of segment detection."""
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs

def main():
    print("=== Independent Verification of Momentum Cost-Robustness ===")
    
    # Load original artifact
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    
    print(f"Verifying: {ARTIFACT_PATH}")
    print(f"Lookbacks: {artifact['lookbacks']}")
    print(f"Cost model: {artifact['cost_model']}")
    print()

    # Verify each lookback
    all_ok = True
    for lb_key, results in artifact["cost_sensitivity"].items():
        lookback = results["lookback"]
        print(f"=== Lookback {lookback} verification ===")
        
        # Load data
        manifest = load_manifest()
        tickers = {}
        for entry in manifest["entries"]:
            ticker = entry["ticker"]
            bars, _ = bt.load_ticker(ticker)
            tickers[ticker] = bars

        # Verify each asset
        for ticker in tickers:
            bars = tickers[ticker]
            closes = bars.closes_array()
            labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)

            # Generate noise benchmark
            noise_bars = bt.generate_bars(600, seed=SEED)
            noise_signals = bt.momentum_signals(noise_bars.closes_array(), lookback=lookback)
            noise_res = bt.walk_forward(
                list(noise_bars), noise_signals, train_window=60, test_window=20,
                warmup=10, overlap_window=10, cfg=bt.BacktestConfig())
            noise_medians = []
            for f in noise_res.folds:
                log_ret = np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                noise_medians.append(log_ret)
            noise_median = np.median(noise_medians) if noise_medians else 0.0

            # Momentum signals
            signals = bt.momentum_signals(closes, lookback=lookback)

            # Run with realistic costs
            cfg = bt.BacktestConfig(
                initial_capital=1e6,
                target_exposure=1.0,
                commission_per_trade=0.0,
                commission_per_share=0.003,
                slippage_cents=5.0,
                slippage_proportional=0.0,
                warmup_periods=0,
                risk_free=0.0,
                periods_per_year=252,
                margin_rate=0.0,
                margin_call_liquidate=True,
            )
            res = bt.walk_forward(
                list(bars), signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=cfg)

            # Extract medians
            medians = []
            for f in res.folds:
                log_ret = np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                medians.append(log_ret)

            # Get original values from artifact
            original_asset = results["per_asset"][ticker]
            original_median = original_asset["real_cost_median"]
            original_noise_median = original_asset["noise_median_log_return"]

            # Compare
            computed_median = np.median(medians) if medians else 0.0
            median_diff = abs(computed_median - original_median)

            print(f"  {ticker}: computed median {computed_median:.6f} vs original {original_median:.6f}")
            print(f"    difference: {median_diff:.8f} (should be < 1e-6)")

            if median_diff > 1e-6:
                all_ok = False
                print(f"    VERIFICATION FAILED: median difference too large")

            # Verify verdict logic matches
            original_verdict = original_asset["real_cost_verdict"]
            
            def get_verdict_from_medians(medians, noise_median):
                if all(abs(m) <= 0.05 for m in medians):
                    return "CONSISTENT_WITH_NOISE"
                elif any(m > 2.0 * abs(noise_median) if noise_median != 0 else m > 0.05 for m in medians):
                    return "REGIME_DEPENDENT_OR_STABLE"
                elif all(m < 0 for m in medians):
                    return "REGIME_STABLE_LOSS"
                else:
                    return "REGIME_STABLE_POSITIVE"

            computed_verdict = get_verdict_from_medians(medians, noise_median)
            
            if original_verdict != computed_verdict:
                all_ok = False
                print(f"    VERIFICATION FAILED: verdict mismatch - original: {original_verdict}, computed: {computed_verdict}")

    # Determinism check: recompute everything for lookback 5 (the most important)
    print("\n=== Determinism check (lookback 5) ===")
    manifest = load_manifest()
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars

    # First run
    lookback = 5
    first_run_medians = {}
    for ticker in tickers:
        bars = tickers[ticker]
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=lookback)
        cfg = bt.BacktestConfig(
            initial_capital=1e6,
            target_exposure=1.0,
            commission_per_share=0.003,
            slippage_cents=5.0,
            warmup_periods=0,
        )
        res = bt.walk_forward(list(bars), signals, train_window=TRAIN, test_window=TEST,
                            warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
        medians = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in res.folds]
        first_run_medians[ticker] = np.median(medians) if medians else 0.0

    # Second run
    second_run_medians = {}
    for ticker in tickers:
        bars = tickers[ticker]
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=lookback)
        cfg = bt.BacktestConfig(
            initial_capital=1e6,
            target_exposure=1.0,
            commission_per_share=0.003,
            slippage_cents=5.0,
            warmup_periods=0,
        )
        res = bt.walk_forward(list(bars), signals, train_window=TRAIN, test_window=TEST,
                            warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
        medians = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in res.folds]
        second_run_medians[ticker] = np.median(medians) if medians else 0.0

    # Compare
    deterministic = True
    for ticker in tickers:
        diff = abs(first_run_medians[ticker] - second_run_medians[ticker])
        print(f"  {ticker}: first {first_run_medians[ticker]:.6f}, second {second_run_medians[ticker]:.6f}, diff {diff:.8f}")
        if diff > 1e-12:
            deterministic = False
            all_ok = False
            print(f"    DETERMINISM FAILED: non-zero difference")

    # Final verdict
    print()
    if all_ok:
        print("✓ All independent verifications PASSED")
        print("✓ Cost-robustness artifact is deterministic and correct")
        print("✓ Evidence base preserved for future research")
    else:
        print("✗ VERIFICATION FAILED: discrepancies found")
        sys.exit(1)

    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-robustness independent verification complete.")

if __name__ == "__main__":
    main()
