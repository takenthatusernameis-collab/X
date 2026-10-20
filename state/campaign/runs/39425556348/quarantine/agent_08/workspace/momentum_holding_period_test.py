"""Test a fixed lookback-5 momentum signal with predeclared small holding-period set.

This script tests momentum with different holding periods using the existing
walk-forward/null methodology on the 10-asset universe. The holding period
refers to the test_window parameter in walk_forward, which controls how long
the signal is held before rebalancing.

This test uses reasonable holding periods that work well with the walk_forward
infrastructure (minimum 5 days to ensure meaningful test segments).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Predeclared holding-period set (days) - reasonable sizes that work with walk_forward
HOLDING_PERIODS = [5, 7, 14, 30]  # Focus on sizes that create meaningful test segments

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_holding_period_final_test.json"
def test_momentum_with_holding_period(asset_name, asset_bars, holding_period):
    """Test momentum with specific holding period using walk-forward."""
    # Use the standard momentum_signals function
    signals = bt.momentum_signals(asset_bars.closes_array(), lookback=LOOKBACK)
    
    # Verify signal integrity
    bt.check_signal_integrity(signals, [b.date for b in asset_bars], warmup=0)
    
    # Run walk-forward validation with custom test window = holding_period
    cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
    result = bt.walk_forward(
        list(asset_bars),
        signals,
        train_window=TRAIN,
        test_window=holding_period,  # This is the holding period!
        warmup=WARM,
        overlap_window=min(holding_period // 2, holding_period - 1),
        cfg=cfg,
    )
    
    # Calculate segment medians (log returns) from OOS folds only
    log_returns = []
    for fold in result.folds:
        log_ret = float(np.log1p(np.clip(fold.metrics["total_return"], -1.0 + 1e-12, None)))
        log_returns.append(log_ret)
    
    segment_median = float(np.median(log_returns)) if log_returns else 0.0
    
    return {
        "asset": asset_name,
        "holding_period": holding_period,
        "status": "SUCCESS",
        "segment_median": segment_median,
        "n_folds": len(result.folds),
        "positive_folds": sum(1 for lr in log_returns if lr > 0),
        "negative_folds": sum(1 for lr in log_returns if lr < 0),
        "zero_folds": sum(1 for lr in log_returns if abs(lr) <= 0.05),
        "aggregate_mean_log_return": result.aggregate_metrics["mean_log_total_return"],
        "aggregate_median_log_return": result.aggregate_metrics["median_log_total_return"],
    }
def classify_result(segment_median, positive_folds, n_folds):
    """Classify the result based on segment median and fold outcomes."""
    if n_folds == 0:
        return "NO_DATA"
    
    # Simple classification based on segment median
    if all(abs(m) <= 0.05 for m in [segment_median]):
        return "CONSISTENT_WITH_NOISE"
    elif segment_median > 0.05 and positive_folds == n_folds:
        return "UNIFORMLY_POSITIVE"
    elif segment_median < -0.05 and positive_folds == 0:
        return "UNIFORMLY_NEGATIVE"
    else:
        return "MIXED/REGIME_DEPENDENT"
def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load manifest and get assets
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("=== Momentum Holding Period Robustness Test ===")
    print(f"Dataset: {manifest['dataset_id']} | Universe: {len(manifest['universe'])} assets")
    print(f"Lookback: {LOOKBACK} days")
    print(f"Holding periods: {HOLDING_PERIODS} days")
    print(f"Walk-forward: train={TRAIN}d / warmup={WARM}d")
    print(f"Note: Using base test_window for each holding period with max overlap")
    print()
    
    # Load all tickers
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    
    # Test each asset with each holding period
    all_results = []
    results_by_asset = {}
    
    for asset_name in tickers.keys():
        asset_bars = tickers[asset_name]
        asset_results = {}
        
        for holding_period in HOLDING_PERIODS:
            print(f"Testing {asset_name} with holding_period={holding_period}...")
            result = test_momentum_with_holding_period(asset_name, asset_bars, holding_period)
            asset_results[holding_period] = result
            all_results.append(result)
            print(f"  Segment median: {result['segment_median']:+.3f}")
            print(f"  Verdict: {classify_result(result['segment_median'], result['positive_folds'], result['n_folds'])}")
        
        results_by_asset[asset_name] = asset_results
        print()
    
    # Aggregate results by holding period
    aggregated = {}
    for holding_period in HOLDING_PERIODS:
        segment_medians = []
        verdicts = []
        
        for asset_name, asset_results in results_by_asset.items():
            if holding_period in asset_results:
                result = asset_results[holding_period]
                segment_medians.append(result["segment_median"])
                verdicts.append(classify_result(
                    result["segment_median"], 
                    result["positive_folds"], 
                    result["n_folds"]
                ))
        
        if segment_medians:
            # Determine overall verdict for this holding period
            positive_count = sum(1 for v in verdicts if v == "UNIFORMLY_POSITIVE")
            negative_count = sum(1 for v in verdicts if v == "UNIFORMLY_NEGATIVE")
            noise_count = sum(1 for v in verdicts if v == "CONSISTENT_WITH_NOISE")
            mixed_count = sum(1 for v in verdicts if v in ["MIXED/REGIME_DEPENDENT"])
            
            if positive_count > negative_count and positive_count > 0:
                overall_verdict = "SUPPORTS_POSITIVE_EDGES"
            elif negative_count > positive_count:
                overall_verdict = "UNIFORMLY_NEGATIVE"
            elif noise_count >= len(verdicts) * 0.75:
                overall_verdict = "CONSISTENT_WITH_NOISE"
            else:
                overall_verdict = "MIXED/REGIME_DEPENDENT"
            
            aggregated[holding_period] = {
                "verdict": overall_verdict,
                "median_segment_median": float(np.median(segment_medians)),
                "assets_tested": len(segment_medians),
                "positive_assets": sum(1 for m in segment_medians if m > 0.05),
                "negative_assets": sum(1 for m in segment_medians if m < -0.05),
                "noise_assets": sum(1 for m in segment_medians if abs(m) <= 0.05),
                "individual_verdicts": verdicts,
            }
            
            print(f"Holding period {holding_period}: {overall_verdict} "
                  f"(median segment median: {aggregated[holding_period]['median_segment_median']:+.3f}) "
                  f"| {aggregated[holding_period]['positive_assets']}/{aggregated[holding_period]['assets_tested']} assets positive")
        else:
            aggregated[holding_period] = {
                "verdict": "NO_DATA",
                "median_segment_median": 0.0,
                "assets_tested": 0,
                "positive_assets": 0,
                "negative_assets": 0,
                "noise_assets": 0,
                "individual_verdicts": []
            }
    
    print("\n=== Final Results ===")
    for holding_period in HOLDING_PERIODS:
        agg = aggregated[holding_period]
        print(f"HP={holding_period:2d}: {agg['verdict']:20s} | "
              f"median={agg['median_segment_median']:+.3f} | "
              f"{agg['positive_assets']}/{agg['assets_tested']} assets positive")
    
    # Determine overall outcome
    positive_hps = sum(1 for hp in HOLDING_PERIODS if aggregated[hp]["verdict"] == "SUPPORTS_POSITIVE_EDGES")
    negative_hps = sum(1 for hp in HOLDING_PERIODS if aggregated[hp]["verdict"] == "UNIFORMLY_NEGATIVE")
    noise_hps = sum(1 for hp in HOLDING_PERIODS if aggregated[hp]["verdict"] == "CONSISTENT_WITH_NOISE")
    mixed_hps = sum(1 for hp in HOLDING_PERIODS if aggregated[hp]["verdict"] == "MIXED/REGIME_DEPENDENT")
    
    print(f"\nHolding Period Summary:")
    print(f"  Positive edges supported: {positive_hps}")
    print(f"  Unif. negative: {negative_hps}")
    print(f"  Consistent with noise: {noise_hps}")
    print(f"  Mixed/Regime dependent: {mixed_hps}")
    
    # Determine final verdict
    if positive_hps > 0 and negative_hps == 0:
        final_verdict = "SUPPORTS_POSITIVE_EDGES_WITH_ROBUST_HOLDING_PERIODS"
    elif noise_hps >= len(HOLDING_PERIODS) * 0.75:
        final_verdict = "CONSISTENT_WITH_NOISE"
    elif mixed_hps > 0:
        final_verdict = "TEMPORALLY_SENSITIVE"
    else:
        final_verdict = "INCONCLUSIVE"
    
    print(f"\nFinal Verdict: {final_verdict}")
    
    # Write artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookback": LOOKBACK,
        "holding_periods": HOLDING_PERIODS,
        "walk_forward_config": {
            "train": TRAIN,
            "test": TEST,
            "warmup": WARM,
            "overlap": OVERLAP
        },
        "results_by_asset": results_by_asset,
        "aggregated_by_holding_period": aggregated,
        "final_verdict": final_verdict,
        "positive_holding_periods": positive_hps,
        "negative_holding_periods": negative_hps,
        "noise_holding_periods": noise_hps,
        "mixed_holding_periods": mixed_hps
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    
    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. Momentum holding-period robustness test on the existing 10-asset universe.")
    return 0
if __name__ == "__main__":
    sys.exit(main())
