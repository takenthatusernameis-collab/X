#!/usr/bin/env python3
"""Cost-sensitivity check for momentum lookback 3/5/10 with realistic costs.

Objective: test whether the momentum edge observed on real data survives
conservative transaction-cost stress.

Method: long the previous lookback-day return; hold 1 day; daily rebalance; 
lookbacks 3,5,10; same walk-forward windows and volatility blocks as 
`research/checks/momentum.py`; each asset tested against a matched null 
(random signal) with identical costs.

Cost model (realistic, per-trade): commission $0.005 per share,
slippage 5 cents per share, proportional slippage 10 bps.

Research / simulation only. No live trading or production execution.
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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_results.json"

# Realistic cost model (per-trade, applied to every round-trip)
COST_CONFIG = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.005,  # $0.005 per share
    slippage_cents=5.0,          # 5 cents per share fixed
    slippage_proportional=0.001, # 10 bps proportional
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)


def matched_null_signals(closes):
    """Random signal class matched to momentum (independent coin-flip null)."""
    n = len(closes)
    np.random.seed(SEED)
    # Coin-flip random weights at each bar, independent of price
    random_weights = np.random.choice([-1.0, 0.0, 1.0], size=n, p=[0.25, 0.5, 0.25])
    return [bt.Signal(date=i + 1, weight=float(random_weights[i])) for i in range(n)]


def get_momentum_cost_artifact():
    """Run the momentum cost check and return the artifact."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Momentum Cost-Sensitivity Check ===")
    print(f"Cost model: commission ${COST_CONFIG.commission_per_share}/share, "
          f"slippage {COST_CONFIG.slippage_cents}c + {COST_CONFIG.slippage_proportional*100:.0f}bps")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Universe: {len(manifest['universe'])} tickers")

    # Load all tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    # Run momentum check with costs for all lookbacks
    results_all = {}
    null_results_all = {}

    for ticker in manifest["universe"]:
        bars = tickers_univ[ticker]
        results_all[ticker] = {}
        null_results_all[ticker] = {}
        
        print(f"  {ticker}: ", end="")
        
        # Generate momentum signals for all lookbacks
        for lb in LOOKBACKS:
            closes = bars.closes_array()
            
            # Momentum signals
            momentum_sig = bt.momentum_signals(closes, lookback=lb)
            
            # Generate momentum result using the same method as momentum_lookback_sweep.py
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            seg_fn = bt.segment_fn_from_labels(labels)
            
            res = bt.stress_segments(
                signals_fn=lambda c, **kw: momentum_sig,
                bars=bars,
                signals=momentum_sig,
                param_grid=[{"lookback": lb}],
                baseline=(("lookback", lb),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
                cfg=COST_CONFIG,
            )
            
            # Extract results
            scen = res.scenarios[0]
            results_all[ticker][str(lb)] = {
                "segments": [s.name for s in res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                "candidate_dispersion": round(res.candidate_dispersion, 3),
                "null_dispersion": round(res.null_dispersion, 3),
                "n_folds": res.n_folds,
                "verdict": res.overall_verdict,
            }
            
            # Generate matched null signals
            null_sig = matched_null_signals(closes)
            
            # Generate null result using the same method
            null_res = bt.stress_segments(
                signals_fn=lambda c, **kw: null_sig,
                bars=bars,
                signals=null_sig,
                param_grid=[{"lookback": lb}],
                baseline=(("lookback", lb),),
                segment_fn=seg_fn,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
                cfg=COST_CONFIG,
            )
            
            # Extract null results
            null_scen = null_res.scenarios[0]
            null_results_all[ticker][str(lb)] = {
                "segments": [s.name for s in null_res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in null_res.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in null_res.scenarios],
                "candidate_dispersion": round(null_res.candidate_dispersion, 3),
                "null_dispersion": round(null_res.null_dispersion, 3),
                "n_folds": null_res.n_folds,
                "verdict": null_res.overall_verdict,
            }
        
        print(f" lookback {LOOKBACKS} completed")

    # Determine cost-robustness verdicts
    print("\n=== Cost-Robustness Analysis ===")
    cost_robust_summary = {}
    
    for ticker in manifest["universe"]:
        edge_counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_DEPENDENT": 0, 
                      "REGIME_STABLE_LOSS": 0, "REGIME_STABLE": 0}
        
        for lb in LOOKBACKS:
            result = results_all[ticker][str(lb)]
            medians = result["medians"]
            
            # Classify edge status
            if all(abs(m) <= 0.05 for m in medians):
                edge_status = "CONSISTENT_WITH_NOISE"
            elif result["candidate_dispersion"] > 2.0 * result["null_dispersion"]:
                edge_status = "REGIME_DEPENDENT"
            elif all(m < 0 for m in medians):
                edge_status = "REGIME_STABLE_LOSS"
            elif all(m > 0 for m in medians):
                edge_status = "REGIME_STABLE"
            else:
                edge_status = "REGIME_STABLE"
            
            edge_counts[edge_status] += 1
        
        # Cost-robustness: REGIME_STABLE at >= 2 lookbacks (conservative threshold)
        cost_robust = "COST_ROBUST" if edge_counts["REGIME_STABLE"] >= 2 else "COST_SENSITIVE"
        cost_robust_summary[ticker] = {
            "edge_counts": edge_counts,
            "cost_robustness_verdict": cost_robust,
            "momentum_results": results_all[ticker],
            "null_results": null_results_all[ticker],
        }
        
        print(f"  {ticker:6s}: {cost_robust} "
              f"(REGIME_STABLE={edge_counts['REGIME_STABLE']}, "
              f"CONSISTENT_WITH_NOISE={edge_counts['CONSISTENT_WITH_NOISE']})")

    # Summary statistics
    robustness_counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0}
    for ticker in cost_robust_summary.values():
        robustness_counts[ticker["cost_robustness_verdict"]] += 1
    
    print(f"\nSummary: {robustness_counts['COST_ROBUST']} COST_ROBUST, "
          f"{robustness_counts['COST_SENSITIVE']} COST_SENSITIVE")

    # Create artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "cost_config": {
            "commission_per_share": COST_CONFIG.commission_per_share,
            "slippage_cents": COST_CONFIG.slippage_cents,
            "slippage_proportional": COST_CONFIG.slippage_proportional,
        },
        "per_asset": cost_robust_summary,
        "robustness_summary": robustness_counts,
    }

    # Write artifact
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n✓ Artifact written to: {ARTIFACT_PATH}")
    return artifact


if __name__ == "__main__":
    get_momentum_cost_artifact()
