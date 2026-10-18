"""Momentum check with explicit cost parameters.

This script runs the momentum check with realistic cost parameters
to test cost robustness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_with_costs_fixed_results.json"

# Realistic transaction cost model
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.003,
    commission_per_share=0.000001,
    slippage_cents=0.05,
    slippage_proportional=0.0005,
    warmup_periods=WARM,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)

# No cost model (baseline)
NO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=WARM,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("=== Momentum Check with Cost Parameters (Fixed) ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Lookback: {LOOKBACK}")
    print(f"\nRealistic costs:")
    print(f"  commission_per_trade: {REALISTIC_COSTS.commission_per_trade}")
    print(f"  commission_per_share: {REALISTIC_COSTS.commission_per_share}")
    print(f"  slippage_cents: {REALISTIC_COSTS.slippage_cents}")
    print(f"  slippage_proportional: {REALISTIC_COSTS.slippage_proportional}")
    
    # Load all tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    
    # Test with realistic costs
    print(f"\n=== Testing momentum with realistic costs ===")
    summary_real = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=LOOKBACK),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", LOOKBACK),),
        param_grid=[{"lookback": LOOKBACK}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=REALISTIC_COSTS,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    # Calculate universe-level candidate dispersion from per-asset results
    all_real_candidate_medians = []
    for ticker in summary_real.ticker_order:
        asset_result = summary_real.assets[ticker]
        all_real_candidate_medians.extend([s.baseline_median_log_return for s in asset_result.scenarios])
    real_candidate_dispersion = float(np.std(all_real_candidate_medians)) if all_real_candidate_medians else 0.0
    
    # Calculate universe-level null dispersion from per-asset results
    all_real_null_medians = []
    for ticker in summary_real.ticker_order:
        asset_result = summary_real.assets[ticker]
        all_real_null_medians.extend([s.noise_median_log_return for s in asset_result.scenarios])
    real_null_dispersion = float(np.std(all_real_null_medians)) if all_real_null_medians else 0.0
    
    print(f"\nRealistic costs results:")
    print(f"  Verdict counts: {summary_real.verdict_counts}")
    print(f"  Candidate dispersion: {real_candidate_dispersion:.3f}")
    print(f"  Null dispersion: {real_null_dispersion:.3f}")
    
    # Test with no costs
    print(f"\n=== Testing momentum with no costs (baseline) ===")
    summary_baseline = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=LOOKBACK),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", LOOKBACK),),
        param_grid=[{"lookback": LOOKBACK}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=NO_COSTS,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    # Calculate universe-level candidate dispersion from per-asset results
    all_baseline_candidate_medians = []
    for ticker in summary_baseline.ticker_order:
        asset_result = summary_baseline.assets[ticker]
        all_baseline_candidate_medians.extend([s.baseline_median_log_return for s in asset_result.scenarios])
    baseline_candidate_dispersion = float(np.std(all_baseline_candidate_medians)) if all_baseline_candidate_medians else 0.0
    
    # Calculate universe-level null dispersion from per-asset results
    all_baseline_null_medians = []
    for ticker in summary_baseline.ticker_order:
        asset_result = summary_baseline.assets[ticker]
        all_baseline_null_medians.extend([s.noise_median_log_return for s in asset_result.scenarios])
    baseline_null_dispersion = float(np.std(all_baseline_null_medians)) if all_baseline_null_medians else 0.0
    
    print(f"\nNo costs results:")
    print(f"  Verdict counts: {summary_baseline.verdict_counts}")
    print(f"  Candidate dispersion: {baseline_candidate_dispersion:.3f}")
    print(f"  Null dispersion: {baseline_null_dispersion:.3f}")
    
    # Compare results
    print(f"\n=== Cost Impact Comparison ===")
    print(f"  Realistic vs No costs:")
    print(f"    Candidate dispersion difference: {real_candidate_dispersion - baseline_candidate_dispersion:.3f}")
    print(f"    Null dispersion difference: {real_null_dispersion - baseline_null_dispersion:.3f}")
    
    # Calculate verdict changes
    verdict_changes = {}
    for verdict in ["REGIME_STABLE", "REGIME_DEPENDENT", "CONSISTENT_WITH_NOISE", "REGIME_STABLE_LOSS"]:
        real_count = summary_real.verdict_counts.get(verdict, 0)
        baseline_count = summary_baseline.verdict_counts.get(verdict, 0)
        if real_count != baseline_count:
            verdict_changes[verdict] = real_count - baseline_count
    
    print(f"    Verdict changes: {verdict_changes}")
    
    # Write artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "lookback": LOOKBACK,
        "cost_model": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        "baseline_cost_model": {
            "commission_per_trade": NO_COSTS.commission_per_trade,
            "commission_per_share": NO_COSTS.commission_per_share,
            "slippage_cents": NO_COSTS.slippage_cents,
            "slippage_proportional": NO_COSTS.slippage_proportional,
        },
        "realistic_costs_summary": {
            "verdict_counts": summary_real.verdict_counts,
            "candidate_dispersion": real_candidate_dispersion,
            "null_dispersion": real_null_dispersion,
        },
        "baseline_summary": {
            "verdict_counts": summary_baseline.verdict_counts,
            "candidate_dispersion": baseline_candidate_dispersion,
            "null_dispersion": baseline_null_dispersion,
        },
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== Artifact written ===")
    print(f"Results saved to: {ARTIFACT_PATH}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
