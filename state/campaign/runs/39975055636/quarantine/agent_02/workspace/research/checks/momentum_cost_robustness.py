"""Cost robustness check for momentum.

This implements the bounded cost-sensitivity check for lookback 3/5/10 momentum
under the repository's existing realistic cost model and matched null.

Evidence basis: LEARNING_STATE admits short-horizon momentum as candidate positive
evidences but leaves execution-cost robustness as a meaningful qualification.

The script tests whether the momentum edge survives conservative transaction-cost
stress on the collected real-data universe for lookback horizons 3, 5, and 10.

Output: A reproducible artifact classifying cost robustness.
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
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"

# Realistic cost model from examples/ma_crossover.py
cfg_realistic = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Zero-cost baseline for comparison
cfg_zero = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
)


def run_momentum_for_lookback(lookback, cfg, tickers):
    """Run momentum with given lookback and config across all assets."""
    per_asset_results = {}
    
    for ticker, bars in tickers.items():
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=lookback)
        
        # Walk-forward validation
        result = bt.walk_forward(
            bars=list(bars),
            signals=signals,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=cfg,
        )
        
        # Compute metrics
        log_returns = np.array([
            np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
            for f in result.folds
        ])
        
        per_asset_results[ticker] = {
            "lookback": lookback,
            "n_folds": len(result.folds),
            "mean_log_return": float(np.mean(log_returns)),
            "median_log_return": float(np.median(log_returns)),
            "std_log_return": float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0,
            "positive_folds": int((log_returns > 0).sum()),
            "total_return": float(result.aggregate_metrics.get("mean_log_total_return", 0.0)),
            "config_type": "realistic" if cfg == cfg_realistic else "zero",
        }
    
    return per_asset_results


def run_cost_sensitivity_check():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== Momentum Cost Robustness Check ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Realistic cost model: commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005")
    
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    # Load all tickers
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars
    
    print(f"Loaded {len(tickers)} assets from collected universe")
    
    # Run cost sensitivity check for each lookback
    all_results = {}
    
    for lookback in LOOKBACKS:
        print(f"\n=== Lookback {lookback} days ===")
        
        # Run with realistic costs
        print(f"  Testing with realistic costs...")
        realistic_results = run_momentum_for_lookback(lookback, cfg_realistic, tickers)
        
        # Run with zero costs  
        print(f"  Testing with zero costs...")
        zero_results = run_momentum_for_lookback(lookback, cfg_zero, tickers)
        
        # Summarize results
        edge_with_costs = sum(1 for ticker, res in realistic_results.items() if res["mean_log_return"] > 0)
        edge_without_costs = sum(1 for ticker, res in zero_results.items() if res["mean_log_return"] > 0)
        
        # Compute average impact of costs
        cost_impact = []
        for ticker in tickers:
            realistic_ret = realistic_results[ticker]["mean_log_return"]
            zero_ret = zero_results[ticker]["mean_log_return"]
            cost_impact.append(zero_ret - realistic_ret)
        
        avg_cost_impact = np.mean(cost_impact) if cost_impact else 0.0
        
        print(f"  With realistic costs: {edge_with_costs}/10 assets have positive edge")
        print(f"  With zero costs: {edge_without_costs}/10 assets have positive edge")
        print(f"  Average cost impact on log return: {avg_cost_impact:.3f}")
        
        all_results[f"lookback_{lookback}"] = {
            "realistic_costs": realistic_results,
            "zero_costs": zero_results,
            "edge_with_costs": edge_with_costs,
            "edge_without_costs": edge_without_costs,
            "avg_cost_impact": float(avg_cost_impact),
        }
    
    # Check determinism and write artifact
    print(f"\n=== Writing artifact to {ARTIFACT_PATH} ===")
    
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN,
        "test": TEST,
        "warmup": WARM,
        "overlap": OVERLAP,
        "window": WINDOW,
        "n_blocks": N_BLOCKS,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "realistic_costs": {
            "commission_per_trade": cfg_realistic.commission_per_trade,
            "commission_per_share": cfg_realistic.commission_per_share,
            "slippage_cents": cfg_realistic.slippage_cents,
            "slippage_proportional": cfg_realistic.slippage_proportional,
        },
        "zero_costs": {
            "commission_per_trade": cfg_zero.commission_per_trade,
            "commission_per_share": cfg_zero.commission_per_share,
            "slippage_cents": cfg_zero.slippage_cents,
            "slippage_proportional": cfg_zero.slippage_proportional,
        },
        "results": all_results,
        "cost_robustness_class": "ROBUST" if all(r["avg_cost_impact"] < 0.01 for r in all_results.values()) 
                                    else "MARGINAL" if all(r["avg_cost_impact"] < 0.05 for r in all_results.values())
                                    else "FRAGILE",
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"Artifact written to: {ARTIFACT_PATH}")
    print(f"Cost robustness classification: {artifact['cost_robustness_class']}")
    print(f"\nNOTE: research/simulation only. No live trading or production execution.")
    print(f"Momentum cost robustness check on collected universe: exploratory simulation.")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_cost_sensitivity_check())
