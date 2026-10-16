"""Independent verification script for momentum cost robustness check.

This script recomputes the cost robustness results from a separate code path
and compares against the artifact produced by momentum_cost_sensitivity.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"


def verify_results():
    """Verify the cost robustness results independently."""
    
    if not ARTIFACT_PATH.exists():
        print(f"ERROR: Artifact not found at {ARTIFACT_PATH}")
        return False
    
    # Load the original artifact
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    
    DATA_ID = artifact["dataset_id"]
    LOOKBACKS = artifact["lookbacks"]
    TRAIN = artifact["train"]
    TEST = artifact["test"]
    WARM = artifact["warmup"]
    OVERLAP = artifact["overlap"]
    WINDOW = artifact["window"]
    N_BLOCKS = artifact["n_blocks"]
    MIN_SEGMENT_BARS = artifact["min_segment_bars"]
    
    print("=== Independent Verification of Cost Robustness Results ===")
    print(f"Dataset: {DATA_ID}")
    print(f"Lookbacks: {LOOKBACKS}")
    print()
    
    # Load manifest for verification
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATA_ID
    
    # Load universe
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
    
    # Verify each asset
    verification_passed = True
    all_verified = True
    
    for ticker in artifact["per_asset"]:
        print(f"Verifying {ticker}...")
        
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)
        
        for lookback in LOOKBACKS:
            # Recompute the results independently
            signals = bt.momentum_signals(closes, lookback=lookback)
            res = bt.stress_segments(
                signals_fn=bt.momentum_signals,
                bars=list(bars),
                signals=signals,
                param_grid=[{"lookback": lookback}],
                baseline=(("lookback", lookback),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
            )
            
            # Generate matched null
            null_signals = bt.random_signals(closes, weight_range=(0.0, 0.0))
            null_res = bt.stress_segments(
                signals_fn=bt.random_signals,
                bars=list(bars),
                signals=null_signals,
                param_grid=[{"lookback": lookback}],
                baseline=(("lookback", lookback),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
            )
            
            # Compute cost robustness independently
            cost_robustness = res.baseline_median_log_return - null_res.baseline_median_log_return
            
            # Get the artifact data
            artifact_data = artifact["per_asset"][ticker][str(lookback)]
            
            # Compare values
            tolerance = 1e-3  # 0.001 tolerance for log returns
            
            candidate_diff = abs(res.baseline_median_log_return - artifact_data["candidate_median_log_return"])
            null_diff = abs(null_res.baseline_median_log_return - artifact_data["null_median_log_return"])
            cost_diff = abs(cost_robustness - artifact_data["cost_robustness"])
            
            # Edge significance check
            edge_sig_artifact = artifact_data["edge_significant"]
            edge_sig_computed = abs(cost_robustness) > max(0.05, 2 * null_res.candidate_dispersion)
            
            if candidate_diff > tolerance or null_diff > tolerance or cost_diff > tolerance or edge_sig_artifact != edge_sig_computed:
                print(f"  lookback {lookback}: VERIFICATION FAILED")
                if candidate_diff > tolerance:
                    print(f"    candidate_median_log_return mismatch: computed={res.baseline_median_log_return:.3f}, artifact={artifact_data['candidate_median_log_return']}")
                if null_diff > tolerance:
                    print(f"    null_median_log_return mismatch: computed={null_res.baseline_median_log_return:.3f}, artifact={artifact_data['null_median_log_return']}")
                if cost_diff > tolerance:
                    print(f"    cost_robustness mismatch: computed={cost_robustness:.3f}, artifact={artifact_data['cost_robustness']}")
                if edge_sig_artifact != edge_sig_computed:
                    print(f"    edge_significant mismatch: computed={edge_sig_computed}, artifact={edge_sig_artifact}")
                verification_passed = False
                all_verified = False
            else:
                print(f"  lookback {lookback}: ✓ VERIFIED")
    
    print()
    
    if verification_passed:
        print("✓ ALL VERIFICATIONS PASSED")
        print("The cost robustness results are independently reproducible.")
    else:
        print("✗ VERIFICATION FAILED")
        print("Some results differ from the artifact.")
    
    return verification_passed


def main():
    if verify_results():
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())