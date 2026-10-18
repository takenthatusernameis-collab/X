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
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_with_costs_results.json"

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

def run_asset(asset, tickers, cfg, costs_label):
    """Run momentum test for a single asset."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    results = {}
    
    # Test momentum signal
    signals = bt.momentum_signals(closes, lookback=LOOKBACK)
    if len(signals) != len(closes):
        raise ValueError(f"{asset} momentum: signals {len(signals)} != bars {len(closes)}")
    
    res = bt.stress_segments(
        signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=LOOKBACK),
        bars=bars,
        signals=signals,
        param_grid=[{"lookback": LOOKBACK}],
        baseline=("lookback", LOOKBACK),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    names = [s.name for s in res.scenarios]
    medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
    results["momentum"] = dict(
        segments=names,
        medians=medians,
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
        costs_label=costs_label,
    )
    
    return results

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("=== Momentum Check with Cost Parameters ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Realistic costs: commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    
    # Load all tickers
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
    
    # Test with realistic costs
    print("\n=== Testing with realistic costs ===")
    realistic_results = {}
    for asset in ["AMZN", "JPM"]:
        res = run_asset(asset, tickers, REALISTIC_COSTS, "REALISTIC")
        realistic_results[asset] = res
        print(f"  {asset}: {res['momentum']['verdict']} "
              f"(candidate median: {res['momentum']['medians']})")
    
    # Test with no costs
    print("\n=== Testing with no costs (baseline) ===")
    baseline_results = {}
    for asset in ["AMZN", "JPM"]:
        res = run_asset(asset, tickers, NO_COSTS, "NO_COSTS")
        baseline_results[asset] = res
        print(f"  {asset}: {res['momentum']['verdict']} "
              f"(candidate median: {res['momentum']['medians']})")
    
    # Compare results
    print("\n=== Cost Impact Comparison ===")
    for asset in ["AMZN", "JPM"]:
        realistic_medians = realistic_results[asset]["momentum"]["medians"]
        baseline_medians = baseline_results[asset]["momentum"]["medians"]
        
        realistic_median = np.mean(realistic_medians)
        baseline_median = np.mean(baseline_medians)
        
        cost_impact = realistic_median - baseline_median
        
        print(f"  {asset}:")
        print(f"    Realistic costs: median={realistic_median:+.3f}, verdict={realistic_results[asset]['momentum']['verdict']}")
        print(f"    No costs (baseline): median={baseline_median:+.3f}, verdict={baseline_results[asset]['momentum']['verdict']}")
        print(f"    Cost impact: {cost_impact:+.3f}")
    
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
        "per_asset_results": {
            "realistic": realistic_results,
            "baseline": baseline_results,
        },
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== Artifact written ===")
    print(f"Results saved to: {ARTIFACT_PATH}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
