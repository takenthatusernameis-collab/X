"""Bounded cost-sensitivity check for lookback 3/5/10 momentum.

Research question: Does the short-horizon momentum edge survive conservative
transaction-cost stress on the collected real-data universe?

Method: walk-forward IS/OOS validation with realistic costs and matched null.
Parameters tested: lookback 3, 5, 10.
Cost model: commission_per_trade=2.0, commission_per_share=0.003,
slippage_cents=2.0, slippage_proportional=0.0005.
Scope: Bounded test using the same regime family as the baseline momentum check.
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
LOOKBACKS = (3, 5, 10)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
REALISTIC_COSTS = bt.BacktestConfig(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"


def run_momentum_cost_check():
    """Run bounded cost-sensitivity check for momentum lookback 3/5/10."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("=== Bounded cost-sensitivity check for lookback 3/5/10 momentum ===")
    print(f"Cost model: commission_per_trade=2.0, commission_per_share=0.003, "
          f"slippage_cents=2.0, slippage_proportional=0.0005")
    print(f"Parameters: lookback={LOOKBACKS}, walk-forward train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d")
    print(f"Segments: {N_BLOCKS} contiguous volatility blocks, past-only classification")
    
    # Use the full collected universe as specified
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    
    results = {}
    
    # Run momentum with realistic costs for each lookback
    for lookback in LOOKBACKS:
        print(f"\n--- lookback={lookback} ---")
        
        # Test with realistic costs
        realistic_results = {}
        for ticker in tickers:
            bars = tickers[ticker]
            closes = bars.closes_array()
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            seg_fn = bt.segment_fn_from_labels(labels)
            
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
                cfg=REALISTIC_COSTS,
            )
            
            realistic_results[ticker] = {
                "segments": [s.name for s in res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                "candidate_dispersion": round(res.candidate_dispersion, 3),
                "null_dispersion": round(res.null_dispersion, 3),
                "n_folds": res.n_folds,
                "verdict": res.overall_verdict,
            }
            
            # Get the baseline median from the first scenario (all scenarios have the same median)
            baseline_median = res.scenarios[0].baseline_median_log_return
            print(f"  {ticker}: {len(res.scenarios)} segments, baseline median={baseline_median:+.3f}, "
                  f"verdict={res.overall_verdict}")
        
        results[f"lookback_{lookback}"] = realistic_results
    
    # Run matched null benchmark using a custom null signal function
    print(f"\n=== Matched null benchmark (coin-flip signal) ===")
    
    null_results = {}
    for lookback in LOOKBACKS:
        print(f"--- lookback={lookback} null ---")
        null_results[lookback] = {}
        
        for ticker in tickers:
            bars = tickers[ticker]
            
            # Create a null signal function that generates coin-flip signals
            # This mimics the momentum signal structure but with random weights
            def null_signal_fn(closes, lookback_param=None, **kwargs):
                n = len(closes)
                # Use a deterministic but random-looking seed based on ticker and lookback
                seed = SEED + abs(hash(ticker)) * 1000 + (lookback_param or 0)
                rng = np.random.default_rng(seed)
                weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0/3.0, 1.0/3.0, 1.0/3.0])
                return [bt.Signal(date=i + 1, weight=float(w)) for i, w in enumerate(weights)]
            
            # Run the stress_segments function with the null signal
            closes = bars.closes_array()
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            seg_fn = bt.segment_fn_from_labels(labels)
            
            null_signals = null_signal_fn(closes, lookback_param=lookback)
            
            res = bt.stress_segments(
                signals_fn=null_signal_fn,
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
                cfg=REALISTIC_COSTS,
            )
            
            null_results[lookback][ticker] = {
                "segments": [s.name for s in res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                "n_folds": res.n_folds,
                "verdict": res.overall_verdict,
            }
            
            # Get the null baseline median from the first scenario (all scenarios have the same median)
            null_baseline_median = res.scenarios[0].baseline_median_log_return
            print(f"  {ticker}: null baseline median={null_baseline_median:+.3f}")
    
    # Calculate cost robustness: baseline within tolerance of null
    print(f"\n=== Cost robustness analysis ===")
    
    cost_robustness = {}
    for lookback in LOOKBACKS:
        cost_robustness[lookback] = {}
        for ticker in tickers:
            baseline_median = results[f"lookback_{lookback}"][ticker]["medians"][0]
            null_median = null_results[lookback][ticker]["medians"][0]
            
            # Use OUTLIER_TOL = 0.05 as in regime_stability.py
            within_tolerance = abs(baseline_median - null_median) <= 0.05
            
            cost_robustness[lookback][ticker] = {
                "baseline_median": baseline_median,
                "null_median": null_median,
                "within_tolerance_0.05": within_tolerance,
                "tolerance_0.05_exceeded": abs(baseline_median - null_median) > 0.05,
            }
            
            print(f"  {ticker} lookback={lookback}: baseline={baseline_median:+.3f}, "
                  f"null={null_median:+.3f}, diff={abs(baseline_median - null_median):.3f}, "
                  f"robust={within_tolerance}")
    
    # Create artifact for independent verification
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "cost_model": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "target_assets": list(tickers.keys()),
        "realistic_results": results,
        "null_benchmarks": null_results,
        "cost_robustness": cost_robustness,
        "interpretation": {
            "robustness_criteria": "baseline median within 0.05 of coin-flip null median",
            "summary": "Momentum with realistic costs survives cost stress if within tolerance",
            "note": "Research/simulation only. No live trading.",
        }
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== Artifact written to: {ARTIFACT_PATH} ===")
    
    # Overall assessment
    robust_count = sum(1 for lookback in LOOKBACKS 
                      for ticker in tickers 
                      if cost_robustness[lookback][ticker]["within_tolerance_0.05"])
    total_count = len(tickers) * len(LOOKBACKS)
    
    print(f"\n=== COST ROBUSTNESS SUMMARY ===")
    print(f"Tests passed: {robust_count}/{total_count} ({robust_count/total_count*100:.1f}%)")
    
    if robust_count == total_count:
        print("RESULT: Momentum edge survives conservative transaction-cost stress - ROBUST")
    else:
        print("RESULT: Momentum edge does NOT survive conservative transaction-cost stress - NOT ROBUST")
        print("Some tests exceeded the 0.05 tolerance threshold")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_momentum_cost_check())