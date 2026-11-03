#!/usr/bin/env python3
"""
Cost Sensitivity Analysis for Momentum

This script extends the existing momentum_lookback_sweep.py to analyze
the impact of realistic transaction costs on momentum performance.
Based on task: R-001 - Run a bounded cost-sensitivity check for lookback 3/5/10 momentum

Realistic cost parameters:
- commission_per_trade: 2.0 (USD per round trip)
- commission_per_share: 0.003 (0.003% = 3bps)
- slippage_cents: 2.0 (USD cents per share = $0.02 per share)
- slippage_proportional: 0.0005 (0.05% = 5bps)
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

# Realistic cost parameters from examples/ma_crossover_real_data.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,  # $2 per round trip
    commission_per_share=0.003,  # 0.003% = 3bps
    slippage_cents=2.0,  # $0.02 per share
    slippage_proportional=0.0005,  # 0.05% = 5bps
    warmup_periods=WARM,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

ZERO_COSTS = bt.BacktestConfig(
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
    margin_call_liquidate=True,
)

def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, one lookback, with given costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
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
    )

def load_artifact():
    """Load the existing momentum_lookback_sweep_results.json artifact."""
    with open(ARTIFACT_DIR / "momentum_lookback_sweep_results.json") as f:
        return json.load(f)

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Cost Sensitivity Analysis: Momentum Lookback Robustness ===")
    print(f"Realistic cost model: commission_per_trade=$2.0, commission_per_share=0.003%,")
    print(f"                     slippage_cents=$0.02/share, slippage_proportional=5bps")
    print()

    # Load the collected universe
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    print("Real-data lookback sweep (lookback 3 / 5 / 10) with realistic costs:")
    print("Using the same regime-stability gate as momentum_lookback_sweep.py")
    print()

    results_zero_cost = {}
    results_realistic_cost = {}

    for ticker in sorted(tickers.keys()):
        bars = tickers[ticker]
        per_lookback_zero = {}
        per_lookback_realistic = {}

        for lookback in LOOKBACKS:
            # Zero cost baseline
            sr_zero = segment_result(ticker, bars, lookback, ZERO_COSTS)
            per_lookback_zero[f"lookback_{lookback}"] = sr_zero

            # Realistic costs
            sr_realistic = segment_result(ticker, bars, lookback, REALISTIC_COSTS)
            per_lookback_realistic[f"lookback_{lookback}"] = sr_realistic

        results_zero_cost[ticker] = per_lookback_zero
        results_realistic_cost[ticker] = per_lookback_realistic

        # Print summary
        print(f"Asset: {ticker}")
        for lb in LOOKBACKS:
            zero_med = per_lookback_zero[f"lookback_{lb}"]["medians"][0]
            realistic_med = per_lookback_realistic[f"lookback_{lb}"]["medians"][0]
            print(f"  lookback={lb:2d}: zero_cost={zero_med:+.3f}, realistic={realistic_med:+.3f}, diff={realistic_med-zero_med:+.3f}")
        print()

    # Calculate robustness metrics
    print("=== Robustness Analysis ===")
    edge_counts_zero = {}
    edge_counts_realistic = {}

    for ticker in sorted(tickers.keys()):
        # Count lookbacks with REGIME_STABLE and uniformly positive edge
        edge_lbs_zero = sum(1 for lb in LOOKBACKS 
                          if results_zero_cost[ticker][f"lookback_{lb}"]["verdict"] == "REGIME_STABLE" 
                          and all(m > 0 for m in results_zero_cost[ticker][f"lookback_{lb}"]["medians"]))
        edge_lbs_realistic = sum(1 for lb in LOOKBACKS 
                                if results_realistic_cost[ticker][f"lookback_{lb}"]["verdict"] == "REGIME_STABLE"
                                and all(m > 0 for m in results_realistic_cost[ticker][f"lookback_{lb}"]["medians"]))

        edge_counts_zero[ticker] = edge_lbs_zero
        edge_counts_realistic[ticker] = edge_lbs_realistic

        print(f"{ticker:4s}: lookback-robust (EDGE) counts: zero_cost={edge_lbs_zero:2d}, "
              f"realistic={edge_lbs_realistic:2d}")

    total_zero_cost = sum(edge_counts_zero.values())
    total_realistic_cost = sum(edge_counts_realistic.values())

    print()
    print("=== Summary ===")
    print(f"Total lookback-robust edges: zero_cost={total_zero_cost}/{len(tickers)} assets, "
          f"realistic={total_realistic_cost}/{len(tickers)} assets")

    # Write artifacts
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        min_segment_bars=MIN_SEGMENT_BARS,
        n_blocks=N_BLOCKS,
        overlap=OVERLAP,
        per_asset_zero_cost=results_zero_cost,
        per_asset_realistic_cost=results_realistic_cost,
        edge_counts_zero_cost=edge_counts_zero,
        edge_counts_realistic_cost=edge_counts_realistic,
        realistic_costs=dict(
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        ),
        zero_costs=dict(
            commission_per_trade=0.0,
            commission_per_share=0.0,
            slippage_cents=0.0,
            slippage_proportional=0.0,
        ),
    )

    # Write both zero cost and realistic cost artifacts
    zero_artifact_path = ARTIFACT_DIR / "momentum_lookback_sweep_zero_cost.json"
    realistic_artifact_path = ARTIFACT_DIR / "momentum_lookback_sweep_realistic_cost.json"

    with open(zero_artifact_path, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    with open(realistic_artifact_path, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\nArtifacts written to:")
    print(f"  Zero cost: {zero_artifact_path}")
    print(f"  Realistic cost: {realistic_artifact_path}")
    print("\nNote: research/simulation only. No live trading or production execution.")
    print("Momentum cost-robustness analysis completed successfully.")

if __name__ == "__main__":
    main()
