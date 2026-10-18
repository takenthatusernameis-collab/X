"""Simple cost robustness check for lookback 3/5/10 momentum.

This test addresses the bottleneck: candidate positive momentum evidence is
short-horizon and mechanical; cost robustness remains decision-relevant.

Objective: Run a bounded cost-sensitivity check for lookback 3/5/10 momentum
under the repository's existing realistic cost model and matched null.

This script directly applies costs to the walk_forward function rather than
using the higher-level stress_segments_across_tickers function.
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
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Realistic transaction cost model (per share in dollars, bps)
REALISTIC_COSTS = {
    "commission_per_trade": 0.003,       # $0.003 per share (approx $3 per round trip)
    "commission_per_share": 0.000001,   # $0.000001 per share (0.1 bps)
    "slippage_cents": 0.05,             # $0.05 fixed slippage per share (5 cents)
    "slippage_proportional": 0.0005,     # 5 bps proportional slippage
}

# Matched null (no cost) for comparison
NO_COSTS = {
    "commission_per_trade": 0.0,
    "commission_per_share": 0.0,
    "slippage_cents": 0.0,
    "slippage_proportional": 0.0,
}

def create_config(costs):
    """Create a BacktestConfig with specified cost parameters."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        **costs,
        warmup_periods=WARM,
        risk_free=0.0,
        periods_per_year=252,
        margin_rate=0.0,
        margin_call_liquidate=False,
    )

def run_single_asset_cost_test(ticker, lookback, costs, costs_label):
    """Run cost sensitivity test for a single asset with a specific lookback."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    # Load the ticker
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    
    # Convert BarSequence to list for walk_forward compatibility
    bars_list = list(bars)
    
    # Generate momentum signals
    signals = bt.momentum_signals(closes, lookback=lookback)
    
    # Create config
    config = create_config(costs)
    
    # Run walk_forward directly on this asset
    res = bt.walk_forward(
        bars_list, signals,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=config)
    
    # Extract log returns from each fold
    log_returns = np.array([
        np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
        for f in res.folds], dtype=np.float64)
    
    # Compute segment medians via volatility blocks
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    segment_fn = bt.segment_fn_from_labels(labels)
    
    segment_medians = []
    for i in range(len(labels)):
        if labels[i] in ["turbulent", "calm"] and i >= MIN_SEGMENT_BARS:
            # Find the segment that contains this index
            start = i
            while start > 0 and labels[start-1] == labels[i]:
                start -= 1
            end = i
            while end < len(labels) and labels[end] == labels[i]:
                end += 1
            
            # Check if segment is long enough
            if end - start >= MIN_SEGMENT_BARS:
                seg_log_returns = log_returns[start:end]
                if len(seg_log_returns) > 0:
                    segment_medians.append(float(np.median(seg_log_returns)))
    
    overall_return = round(float(np.sum(log_returns)), 3) if len(log_returns) > 0 else 0.0
    
    return {
        "ticker": ticker,
        "lookback": lookback,
        "costs_label": costs_label,
        "medians": [round(m, 3) for m in segment_medians],
        "candidate_dispersion": round(float(np.std(segment_medians)), 3) if segment_medians else 0.0,
        "n_folds": len(res.folds),
        "overall_return": overall_return,
    }

def main():
    print("=== Simple Cost Robustness Check for Momentum (lookback 3/5/10) ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Universe: {len(load_manifest()['universe'])} assets")
    print(f"Realistic costs: {REALISTIC_COSTS}")
    
    manifest = load_manifest()
    all_results = {}
    
    # Test each lookback under realistic costs
    realistic_results = {}
    for lookback in LOOKBACKS:
        print(f"\n=== Testing lookback {lookback} with realistic costs ===")
        lookback_results = []
        for ticker in manifest["universe"]:
            result = run_single_asset_cost_test(ticker, lookback, REALISTIC_COSTS, "REALISTIC")
            lookback_results.append(result)
            median_val = round(float(np.median(result['medians'])) if result['medians'] else 0, 3)
            print(f"  {ticker}: median={median_val:+.3f}")
        
        realistic_results[f"lookback_{lookback}_realistic"] = lookback_results
        all_results[f"lookback_{lookback}_realistic"] = lookback_results
    
    # Test each lookback under no costs
    print(f"\n=== Testing lookback 5 with no costs (baseline) ===")
    baseline_results = []
    for ticker in manifest["universe"]:
        result = run_single_asset_cost_test(ticker, 5, NO_COSTS, "NO_COSTS")
        baseline_results.append(result)
        median_val = round(float(np.median(result['medians'])) if result['medians'] else 0, 3)
        print(f"  {ticker}: median={median_val:+.3f}")
    
    all_results["lookback_5_nocosts"] = baseline_results
    
    # Compute summary statistics
    print(f"\n=== Cost Impact Summary ===")
    
    for lookback in LOOKBACKS:
        realistic_data = realistic_results[f"lookback_{lookback}_realistic"]
        baseline_data = baseline_results
        
        # Calculate overall median log returns
        realistic_medians = [r['medians'] for r in realistic_data if r['medians']]
        baseline_medians = [r['medians'] for r in baseline_data if r['medians']]
        
        realistic_overall_median = round(float(np.median([np.median(m) for m in realistic_medians])), 3) if realistic_medians else 0.0
        baseline_overall_median = round(float(np.median([np.median(m) for m in baseline_medians])), 3) if baseline_medians else 0.0
        
        cost_impact = realistic_overall_median - baseline_overall_median
        
        print(f"Lookback {lookback}:")
        print(f"  Realistic costs: median={realistic_overall_median:+.3f}")
        print(f"  No costs (baseline): median={baseline_overall_median:+.3f}")
        print(f"  Cost impact: {cost_impact:+.3f}")
        
        # Determine if edge survives costs
        edge_survives = realistic_overall_median > 0.0
        print(f"  Edge survives costs? {edge_survives}")
    
    # Write artifact
    artifact_dir = Path.cwd() / "state" / "check_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "cost_model": REALISTIC_COSTS,
        "baseline_cost_model": NO_COSTS,
        "lookback_periods": LOOKBACKS,
        "results_by_lookback": all_results,
    }
    
    artifact_path = artifact_dir / "cost_robustness_simple_results.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== Artifact written ===")
    print(f"Results saved to: {artifact_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
