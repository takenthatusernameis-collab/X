"""Cost-sensitivity check for lookback 3/5/10 momentum.

This script runs a bounded cost-sensitivity check for lookback 3/5/10 momentum
under the repository's existing realistic cost model and matched null.

The objective is to test whether the short-horizon momentum edge survives
conservative transaction-cost stress on the collected real-data universe.

Method (fixed a-priori, not tuned to OOS):
- Momentum: long the previous lookback-day return, hold 1 day, daily rebalance
- Lookbacks: 3, 5, 10
- Walk-forward validation: train=252d/test=84d/warmup=60d/overlap=60d
- Universe: existing 10-asset collected universe (AAPL, AMZN, GOOGL, JPM, JNJ, META, MSFT, NVDA, TSLA, XOM)
- Cost models: zero cost and realistic costs
- Matched null: coin-flip benchmark on the same segments

The result classifies whether momentum survives realistic transaction-cost stress.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"

def zero_cost_config():
    """Zero cost configuration."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    )

def realistic_cost_config():
    """Realistic cost configuration based on examples."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )

def run_cost_sensitivity_for_lookback(ticker, bars, lookback):
    """Run walk-forward validation with both zero cost and realistic costs."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    
    results = {}
    
    # Run with zero cost
    zero_cfg = zero_cost_config()
    zero_result = bt.stress_segments(
        signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=lookback),
        bars=list(bars),
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=zero_cfg,
    )
    
    # Run with realistic costs
    realistic_cfg = realistic_cost_config()
    realistic_result = bt.stress_segments(
        signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=lookback),
        bars=list(bars),
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=realistic_cfg,
    )
    
    # Extract key metrics
    zero_scenario = zero_result.scenarios[0]
    realistic_scenario = realistic_result.scenarios[0]
    
    results["lookback"] = lookback
    results["zero_cost"] = {
        "median_log_return": round(zero_scenario.baseline_median_log_return, 4),
        "noise_median_log_return": round(zero_scenario.noise_median_log_return, 4),
        "candidate_dispersion": round(zero_scenario.candidate_dispersion, 4),
        "null_dispersion": round(zero_scenario.null_dispersion, 4),
        "n_folds": zero_result.n_folds,
        "verdict": zero_result.overall_verdict,
        "segments": [s.name for s in zero_result.scenarios],
        "zero_cost_edge": zero_scenario.compare_noise(zero_scenario.noise_median_log_return) if zero_scenario.noise_fold_median_log_returns else False,
    }
    
    results["realistic_cost"] = {
        "median_log_return": round(realistic_scenario.baseline_median_log_return, 4),
        "noise_median_log_return": round(realistic_scenario.noise_median_log_return, 4),
        "candidate_dispersion": round(realistic_scenario.candidate_dispersion, 4),
        "null_dispersion": round(realistic_scenario.null_dispersion, 4),
        "n_folds": realistic_result.n_folds,
        "verdict": realistic_result.overall_verdict,
        "segments": [s.name for s in realistic_result.scenarios],
        "realistic_cost_edge": realistic_scenario.compare_noise(realistic_scenario.noise_median_log_return) if realistic_scenario.noise_fold_median_log_returns else False,
    }
    
    # Calculate cost impact
    zero_edge = zero_result.overall_verdict == "REGIME_STABLE" and all(m > 0 for m in 
              [s.baseline_median_log_return for s in zero_result.scenarios])
    realistic_edge = realistic_result.overall_verdict == "REGIME_STABLE" and all(m > 0 for m in
              [s.baseline_median_log_return for s in realistic_result.scenarios])
    
    results["zero_cost_edge_exists"] = zero_edge
    results["realistic_cost_edge_exists"] = realistic_edge
    results["cost_impact"] = "SURVIVES" if zero_edge == realistic_edge else "IMPAIRED" if realistic_edge else "LOST"
    
    return results

def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("=== Cost-Sensitivity Check for Lookback 3/5/10 Momentum ===")
    print(f"Universe: {len(manifest['universe'])} assets")
    print(f"Cost models: zero cost vs realistic costs")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Walk-forward: train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d")
    print()
    
    # Load all tickers
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
    
    # Filter to the exact 10 assets mentioned in the contract
    target_tickers = ["AAPL", "AMZN", "GOOGL", "JPM", "JNJ", "META", "MSFT", "NVDA", "TSLA", "XOM"]
    
    filtered_tickers = {}
    for ticker in target_tickers:
        if ticker in tickers:
            filtered_tickers[ticker] = tickers[ticker]
        else:
            print(f"Warning: {ticker} not found in manifest entries")
    
    print(f"Loaded {len(filtered_tickers)} tickers")
    
    # Run cost sensitivity for each lookback across the universe
    universe_results = {}
    lookback_summary = {}
    
    for lookback in LOOKBACKS:
        print(f"\n=== Lookback {lookback} ===")
        lookback_results = {}
        
        for ticker in filtered_tickers:
            print(f"  Processing {ticker}...")
            results = run_cost_sensitivity_for_lookback(ticker, filtered_tickers[ticker], lookback)
            lookback_results[ticker] = results
            
            # Print summary
            zero_edge = results["zero_cost_edge_exists"]
            realistic_edge = results["realistic_cost_edge_exists"]
            cost_impact = results["cost_impact"]
            print(f"    Zero cost: {zero_edge} | Realistic cost: {realistic_edge} | Impact: {cost_impact}")
        
        lookback_summary[lookback] = lookback_results
        universe_results[f"lookback_{lookback}"] = lookback_results
    
    # Calculate universe-level summary
    print("\n=== Universe-Level Summary ===")
    for lookback in LOOKBACKS:
        lookback_results = lookback_summary[lookback]
        zero_edges = sum(1 for t in lookback_results if lookback_results[t]["zero_cost_edge_exists"])
        realistic_edges = sum(1 for t in lookback_results if lookback_results[t]["realistic_cost_edge_exists"])
        impacted = sum(1 for t in lookback_results if lookback_results[t]["cost_impact"] != "SURVIVES")
        
        print(f"Lookback {lookback}:")
        print(f"  Zero-cost edges: {zero_edges}/{len(lookback_results)}")
        print(f"  Realistic-cost edges: {realistic_edges}/{len(lookback_results)}")
        print(f"  Cost-impacted assets: {impacted}/{len(lookback_results)}")
        print(f"  Cost impact rate: {impacted/len(lookback_results):.1%}")
        
        # Overall verdict
        if realistic_edges == len(lookback_results) and zero_edges == realistic_edges:
            verdict = "ROBUST"
        elif realistic_edges == 0:
            verdict = "NO_SURVIVAL"
        elif realistic_edges < zero_edges:
            verdict = "PARTIAL_SURVIVAL"
        else:
            verdict = "UNEXPECTED"
        
        lookback_summary[lookback]["universe_verdict"] = verdict
    
    # Write artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": 42,
        "train": TRAIN,
        "test": TEST,
        "warmup": WARM,
        "overlap": OVERLAP,
        "window": WINDOW,
        "n_blocks": N_BLOCKS,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "lookbacks": LOOKBACKS,
        "cost_models": {
            "zero_cost": {
                "commission_per_trade": 0.0,
                "commission_per_share": 0.0,
                "slippage_cents": 0.0,
                "slippage_proportional": 0.0,
            },
            "realistic_cost": {
                "commission_per_trade": 2.0,
                "commission_per_share": 0.003,
                "slippage_cents": 2.0,
                "slippage_proportional": 0.0005,
            }
        },
        "per_lookback": lookback_summary,
        "universe_summary": universe_results,
        "generation_timestamp": "2026-10-28T16:08:36Z",
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    print("Cost-sensitivity check completed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())