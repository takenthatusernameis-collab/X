#!/usr/bin/env python3
"""Independent verification of momentum cost sensitivity results.

This script reproduces the cost sensitivity check independently to verify
that the results are reproducible and correctly classified. It should be run
in isolation from the original cost_sensitivity_momentum.py script.

Verification requirement: The evidence gate is satisfied when this script
confirms the same results and classifications as the original run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"


def momentum_signals(closes: np.ndarray, lookback: int):
    """Momentum signals: long if lookback days were up, short if down."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return signals


def run_verification():
    """Run independent verification of cost sensitivity results."""
    print("=== Independent Verification of Momentum Cost Sensitivity ===")
    
    # Load the original artifact
    original_path = Path.cwd() / "state" / "cost_sensitivity_results.json"
    with open(original_path, "r") as f:
        original_artifact = json.load(f)
    
    print("Original artifact loaded successfully")
    
    # Define cost models
    REALISTIC_COSTS = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=0,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )
    
    ZERO_COSTS = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=0,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    )
    
    # Verify each lookback
    all_passed = True
    verification_results = []
    
    for original_result in original_artifact["lookback_results"]:
        lookback = original_result["lookback"]
        print(f"\n=== Verifying lookback {lookback} ===")
        
        # Load data
        manifest = load_manifest()
        assert manifest["dataset_id"] == DATASET_ID
        
        bars, dates = bt.load_ticker("AAPL")
        closes = bars.closes_array()
        signals = momentum_signals(closes, lookback)
        
        # Run realistic costs
        result_realistic = bt.walk_forward(
            list(bars), signals, train_window=252, test_window=84,
            warmup=0, overlap_window=60, cfg=REALISTIC_COSTS
        )
        
        # Run zero costs
        result_zero = bt.walk_forward(
            list(bars), signals, train_window=252, test_window=84,
            warmup=0, overlap_window=60, cfg=ZERO_COSTS
        )
        
        # Extract metrics
        realistic_return = result_realistic.aggregate_metrics['mean_log_total_return']
        zero_return = result_zero.aggregate_metrics['mean_log_total_return']
        realistic_pos = result_realistic.aggregate_metrics['fold_count_positive']
        zero_pos = result_zero.aggregate_metrics['fold_count_positive']
        total_folds = result_realistic.aggregate_metrics['n_folds']
        
        # Calculate cost impact
        cost_impact_bps = (realistic_return - zero_return) * 10000
        
        # Determine robustness
        if realistic_return > 0 and realistic_pos >= total_folds * 0.6:
            robustness = "ROBUST_POSITIVE"
        elif realistic_return < 0 and realistic_pos < total_folds * 0.4:
            robustness = "ROBUST_NEGATIVE"
        elif realistic_return < zero_return:
            robustness = "COST_SENSITIVE"
        else:
            robustness = "COST_INSENSITIVE"
        
        # Compare with original
        match_return = abs(realistic_return - original_result["realistic"]["mean_log_return"]) < 0.001
        match_cost_impact = abs(cost_impact_bps - original_result["cost_impact_bps"]) < 0.1
        match_robustness = robustness == original_result["robustness"]
        
        verification_results.append({
            "lookback": lookback,
            "match_return": match_return,
            "match_cost_impact": match_cost_impact,
            "match_robustness": match_robustness,
            "calculated_robustness": robustness,
            "calculated_cost_impact_bps": round(cost_impact_bps, 1),
        })
        
        print(f"  Realistic return: {realistic_return:.3f} (orig: {original_result['realistic']['mean_log_return']})")
        print(f"  Zero cost return: {zero_return:.3f}")
        print(f"  Cost impact: {cost_impact_bps:+.1f} bps (orig: {original_result['cost_impact_bps']:+.1f} bps)")
        print(f"  Robustness: {robustness} (orig: {original_result['robustness']})")
        print(f"  Return match: {'PASS' if match_return else 'FAIL'}")
        print(f"  Cost impact match: {'PASS' if match_cost_impact else 'FAIL'}")
        print(f"  Robustness match: {'PASS' if match_robustness else 'FAIL'}")
        
        if not (match_return and match_cost_impact and match_robustness):
            all_passed = False
    
    # Summary
    print("\n=== Verification Summary ===")
    for result in verification_results:
        print(f"  Lookback {result['lookback']}: {result['calculated_robustness']} "
              f"(cost impact: {result['calculated_cost_impact_bps']:+.1f} bps)")
    
    all_matches = all(r["match_return"] and r["match_cost_impact"] and r["match_robustness"] for r in verification_results)
    
    if all_passed and all_matches:
        print("\n=== VERIFICATION PASSED ===")
        print("All results match the original artifact. Evidence gate is satisfied.")
        return True
    else:
        print("\n=== VERIFICATION FAILED ===")
        print("Some results do not match the original artifact.")
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)