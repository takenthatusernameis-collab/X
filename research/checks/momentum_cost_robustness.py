#!/usr/bin/env python3
"""Run bounded cost-sensitivity check for lookback 3/5/10 momentum with realistic costs.

Objective: Does the short-horizon momentum edge survive conservative transaction-cost stress
on the collected real-data universe?

Uses the existing 10-asset collected universe and existing momentum implementation.
Realistic cost model from examples/ma_crossover.py:
- commission_per_trade=2.0
- commission_per_share=0.003
- slippage_cents=2.0
- slippage_proportional=0.0005

Methodology:
1. Load collected universe from manifest
2. Run momentum checks with realistic costs for lookback 3, 5, 10
3. Compute cost-adjusted performance metrics
4. Compare to matched null with identical costs
5. Determine if edges survive cost stress
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path so we can import research.backtest
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
REALISTIC_COSTS = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ZERO_COSTS = dict(
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
)


def run_momentum_with_costs(ticker, bars, lookback, cost_config):
    """Run momentum analysis for one asset with given cost configuration."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    
    cfg = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        **cost_config
    )
    
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
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
        cfg=cfg,
    )
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
        cost_config=cost_config,
    )


def run_cost_sensitivity_check():
    """Run momentum checks with realistic costs for all lookbacks."""
    ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
    ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("=== Cost Sensitivity Check for Momentum Lookback 3/5/10 ===")
    print(f"Zero cost model: {ZERO_COSTS}")
    print(f"Realistic cost model: {REALISTIC_COSTS}")
    print(f"Universe: {len(manifest['universe'])} assets")
    print()
    
    # Load all tickers
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())
    
    # Run with zero costs
    print("=== Running with zero costs ===")
    zero_cost_results = {lb: {t: run_momentum_with_costs(t, all_tickers[t], lb, ZERO_COSTS) for t in ticker_order}
                        for lb in LOOKBACKS}
    
    # Run with realistic costs
    print("\n=== Running with realistic costs ===")
    realistic_cost_results = {lb: {t: run_momentum_with_costs(t, all_tickers[t], lb, REALISTIC_COSTS) for t in ticker_order}
                              for lb in LOOKBACKS}
    
    # Compare results
    print("\n=== Cost Impact Analysis ===")
    for lb in LOOKBACKS:
        print(f"\nLookback {lb}:")
        cost_impact = []
        for ticker in ticker_order:
            zero_median = zero_cost_results[lb][ticker]["medians"][0]
            realistic_median = realistic_cost_results[lb][ticker]["medians"][0]
            impact = (realistic_median - zero_median) / abs(zero_median) * 100 if zero_median != 0 else float('inf')
            cost_impact.append(impact)
        
        avg_impact = np.mean(cost_impact)
        max_impact = np.max(cost_impact)
        min_impact = np.min(cost_impact)
        
        print(f"  Average cost impact: {avg_impact:+.1f}%")
        print(f"  Range: {min_impact:+.1f}% to {max_impact:+.1f}%")
    
    # Determine robustness with realistic costs
    print("\n=== Robustness Analysis with Realistic Costs ===")
    per_asset_robustness = {}
    robust_assets = []
    sensitive_assets = []
    no_edge_assets = []
    
    for ticker in ticker_order:
        edge_status = {}
        for lb in LOOKBACKS:
            result = realistic_cost_results[lb][ticker]
            edge_status[lb] = "EDGE" if (result["verdict"] == "REGIME_STABLE" and all(m > 0 for m in result["medians"])) else "NO_EDGE"
        
        edge_counts = {lb: edge_status[lb] for lb in LOOKBACKS}
        edge_count = sum(1 for lb in LOOKBACKS if edge_status[lb] == "EDGE")
        
        if edge_count >= 2:
            verdict = "ROBUST"
            robust_assets.append(ticker)
        elif edge_count == 1:
            verdict = "SENSITIVE"
            sensitive_assets.append(ticker)
        else:
            verdict = "NO_EDGE"
            no_edge_assets.append(ticker)
        
        per_asset_robustness[ticker] = {
            "lookback_performance": {
                lb: realistic_cost_results[lb][ticker]["medians"][0] 
                for lb in LOOKBACKS
            },
            "edge_status": edge_status,
            "edge_counts": edge_counts,
            "robustness_verdict": verdict,
        }
        
        print(f"{ticker}: {verdict} (edge at {[lb for lb in LOOKBACKS if edge_status[lb] == 'EDGE']})")
    
    # Overall universe summary
    robust_count = len(robust_assets)
    sensitive_count = len(sensitive_assets)
    no_edge_count = len(no_edge_assets)
    
    print(f"\nUniverse Summary with Realistic Costs:")
    print(f"ROBUST (>=2 lookbacks): {robust_count}")
    print(f"SENSITIVE (exactly 1 lookback): {sensitive_count}")
    print(f"NO_EDGE: {no_edge_count}")
    
    # Cost survival determination
    edge_survives = robust_count > 0
    print(f"\n=== Conclusion ===")
    print(f"Momentum edge survives cost stress: {edge_survives}")
    if edge_survives:
        print(f"Robust assets: {', '.join(robust_assets)}")
        print(f"Cost reduction impact: {np.mean([cost_impact for ticker in ticker_order for cost_impact in []]):+.1f}%")
    else:
        print("No assets showed robust positive edges with realistic costs")
    
    # Write artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        zero_cost=ZERO_COSTS,
        realistic_cost=REALISTIC_COSTS,
        per_asset=per_asset_robustness,
        universe=dict(
            robust_count=robust_count,
            sensitive_count=sensitive_count,
            no_edge_count=no_edge_count,
            robust_assets=robust_assets,
            sensitive_assets=sensitive_assets,
            no_edge_assets=no_edge_assets,
        ),
        edge_survives=edge_survives,
    )
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(run_cost_sensitivity_check())