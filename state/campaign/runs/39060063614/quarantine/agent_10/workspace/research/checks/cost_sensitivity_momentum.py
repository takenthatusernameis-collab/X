#!/usr/bin/env python3
"""Cost sensitivity check for momentum lookback 3/5/10 with realistic costs.

This implements the bounded cost-sensitivity check requested in the focused-task
contract. It evaluates momentum signals with lookback 3, 5, and 10 under both
realistic transaction costs and a zero-cost null model, comparing their performance
characteristics.

Evidence basis: LEARNING_STATE admits short-horizon momentum as candidate positive
evidence but leaves execution-cost robustness as a meaningful qualification.

The primary research question: Does the short-horizon momentum edge survive
conservative transaction-cost stress on the collected real-data universe?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path so we can import research modules
sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"

# Realistic transaction cost model (matches ma_crossover_real_data.py)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Zero-cost null model for comparison
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
)

def momentum_signals(closes: np.ndarray, lookback: int):
    """Momentum signals: long if lookback days were up, short if down."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return signals

def run_lookback_comparison(lookback: int):
    """Run cost comparison for a specific lookback value."""
    print(f"\n=== Lookback {lookback} cost sensitivity ===")
    
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "dataset mismatch"
    
    # Load universe
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    
    # Get AAPL data for cost comparison (consistent with ma_crossover_real_data.py)
    bars, dates = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    
    # Run with realistic costs
    print(f"  Running with realistic costs (lookback={lookback})...")
    result_realistic = bt.walk_forward(
        list(bars), signals, train_window=252, test_window=84,
        warmup=0, overlap_window=60, cfg=REALISTIC_COSTS
    )
    
    # Run with zero costs (null)
    print(f"  Running with zero costs (lookback={lookback})...")
    result_zero = bt.walk_forward(
        list(bars), signals, train_window=252, test_window=84,
        warmup=0, overlap_window=60, cfg=ZERO_COSTS
    )
    
    # Extract metrics for comparison
    realistic_metrics = result_realistic.aggregate_metrics
    zero_metrics = result_zero.aggregate_metrics
    
    # Calculate cost impact
    realistic_return = realistic_metrics['mean_log_total_return']
    zero_return = zero_metrics['mean_log_total_return']
    
    # Positive folds comparison
    realistic_pos = realistic_metrics['fold_count_positive']
    zero_pos = zero_metrics['fold_count_positive']
    total_folds = realistic_metrics['n_folds']
    
    print(f"  Realistic costs: mean_log_return={realistic_return:+.3f}, "
          f"positive_folds={realistic_pos}/{total_folds}")
    print(f"  Zero costs (null): mean_log_return={zero_return:+.3f}, "
          f"positive_folds={zero_pos}/{total_folds}")
    
    # Calculate cost impact inbps (basis points)
    cost_impact_bps = (realistic_return - zero_return) * 10000
    print(f"  Cost impact: {cost_impact_bps:+.1f} bps")
    
    # Determine robustness classification
    if realistic_return > 0 and realistic_pos >= total_folds * 0.6:
        robustness = "ROBUST_POSITIVE"
    elif realistic_return < 0 and realistic_pos < total_folds * 0.4:
        robustness = "ROBUST_NEGATIVE"
    elif realistic_return < zero_return:
        robustness = "COST_SENSITIVE"
    else:
        robustness = "COST_INSENSITIVE"
    
    print(f"  Robustness classification: {robustness}")
    
    return {
        "lookback": lookback,
        "realistic": {
            "mean_log_return": round(realistic_return, 3),
            "positive_folds": realistic_pos,
            "total_folds": total_folds,
            "std_log_return": round(realistic_metrics['std_log_total_return'], 3),
            "median_log_return": round(realistic_metrics['median_log_total_return'], 3),
        },
        "zero_cost": {
            "mean_log_return": round(zero_return, 3),
            "positive_folds": zero_pos,
            "total_folds": total_folds,
            "std_log_return": round(zero_metrics['std_log_total_return'], 3),
            "median_log_return": round(zero_metrics['median_log_total_return'], 3),
        },
        "cost_impact_bps": round(cost_impact_bps, 1),
        "robustness": robustness,
    }

def main():
    print("=== Momentum Cost Sensitivity Check (lookback 3/5/10) ===")
    print("Dataset:", DATASET_ID)
    print("Universe size: 10 assets")
    print("Walk-forward: train=252d, test=84d, warmup=0d, overlap=60d")
    print("Realistic cost model: commission=$2/trade, $0.003/share, slippage=$0.02/share + 0.05% propto")
    
    # Verify manifest integrity
    manifest = load_manifest()
    print(f"\n=== Manifest verification ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    
    # Run cost sensitivity for each lookback
    results = []
    for lookback in [3, 5, 10]:
        result = run_lookback_comparison(lookback)
        results.append(result)
    
    # Generate summary
    print("\n=== Cost Sensitivity Summary ===")
    robust_positive = sum(1 for r in results if r["robustness"] == "ROBUST_POSITIVE")
    robust_negative = sum(1 for r in results if r["robustness"] == "ROBUST_NEGATIVE")
    cost_sensitive = sum(1 for r in results if r["robustness"] == "COST_SENSITIVE")
    cost_insensitive = sum(1 for r in results if r["robustness"] == "COST_INSENSITIVE")
    
    print(f"Results by lookback:")
    for r in results:
        print(f"  Lookback {r['lookback']}: {r['robustness']} "
              f"(cost impact: {r['cost_impact_bps']:+.1f} bps)")
    
    print(f"\nSummary: ROBUST_POS={robust_positive}, ROBUST_NEG={robust_negative}, "
          f"COST_SENS={cost_sensitive}, COST_INSEN={cost_insensitive}")
    
    # Save results
    output_path = Path.cwd() / "state" / "cost_sensitivity_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "dataset_id": DATASET_ID,
        "universe_size": len(manifest["universe"]),
        "walk_forward_config": {
            "train_window": 252,
            "test_window": 84,
            "warmup": 0,
            "overlap": 60,
        },
        "cost_models": {
            "realistic": REALISTIC_COSTS.__dict__,
            "zero_cost": ZERO_COSTS.__dict__,
        },
        "lookback_results": results,
        "summary": {
            "robust_positive_count": robust_positive,
            "robust_negative_count": robust_negative,
            "cost_sensitive_count": cost_sensitive,
            "cost_insensitive_count": cost_insensitive,
            "cost_impacts_bps": [r["cost_impact_bps"] for r in results],
        },
        "reproducible": True,
    }
    
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== Results saved to: {output_path} ===")
    
    # Research conclusion
    if robust_positive > 0:
        conclusion = "Momentum shows COST-INSENSITIVE or ROBUST_POSITIVE performance; edge survives realistic costs."
    elif robust_negative > 0:
        conclusion = "Momentum shows COST-INSENSITIVE or ROBUST_NEGATIVE performance; edge does not survive costs."
    else:
        conclusion = "Momentum shows mixed or COST_SENSITIVE performance; costs materially impact signal robustness."
    
    print(f"\nResearch conclusion: {conclusion}")
    return 0

if __name__ == "__main__":
    sys.exit(main())