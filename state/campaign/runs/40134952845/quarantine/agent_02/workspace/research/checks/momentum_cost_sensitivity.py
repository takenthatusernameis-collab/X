"""Cost sensitivity check for lookback 3/5/10 momentum.

Objective: Test whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected real-data universe.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day return;
hold 1 day; daily rebalance; lookbacks 3, 5, 10; same walk-forward
windows and volatility blocks as `momentum.py`. The realistic cost model is
applied to both the momentum signal and to a matched null (coin-flip) benchmark.

The "matched null" is a coin-flip signal that has the same trading frequency
and cost structure as momentum, but no edge. This allows us to test whether
the momentum edge is distinguishable from random noise under realistic costs.

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

# Realistic cost parameters (from examples/ma_crossover.py)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=2.0,        # $2 per trade
    commission_per_share=0.003,      # $0.003 per share
    slippage_cents=2.0,              # $0.02 per share fixed
    slippage_proportional=0.0005,   # 0.05% proportional
)

ZERO_COSTS = bt.BacktestConfig(initial_capital=1e6, warmup_periods=0)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def load_universe():
    """Load the collected real-data universe."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars
    
    return tickers


def run_cost_sensitivity_test(tickers):
    """Run momentum with realistic costs vs matched null for each lookback."""
    print("=== Loading universe ===")
    print(f"  dataset: {DATASET_ID} | universe: {len(tickers)} assets")
    
    print("\n=== Cost sensitivity analysis ===")
    print("Testing momentum lookback {} with realistic transaction costs:".format(LOOKBACKS))
    print("  Realistic costs: commission_per_trade=$2, commission_per_share=$0.003")
    print("  Realistic costs: slippage_cents=$0.02, slippage_proportional=0.05%")
    print("  Matched null: coin-flip signal with same cost structure")
    
    results = {}
    
    for lookback in LOOKBACKS:
        print(f"\n--- Lookback {lookback} ---")
        
        # Generate momentum signals
        momentum_results = {}
        
        for ticker, bars in tickers.items():
            closes = bars.closes_array()
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            seg_fn = bt.segment_fn_from_labels(labels)
            
            # Generate momentum signals once
            signals = bt.momentum_signals(closes, lookback=lookback)
            
            # Test with realistic costs
            momentum_res = bt.stress_segments(
                signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=lookback),
                bars=bars,
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
            
            momentum_results[ticker] = {
                "segments": [s.name for s in momentum_res.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in momentum_res.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in momentum_res.scenarios],
                "candidate_dispersion": round(momentum_res.candidate_dispersion, 3),
                "null_dispersion": round(momentum_res.null_dispersion, 3),
                "verdict": momentum_res.overall_verdict,
                "n_folds": momentum_res.n_folds,
            }
        
        results[f"lookback_{lookback}"] = momentum_results
        
        print(f"  lookback={lookback}: tested {len(momentum_results)} assets")
        print(f"    verdict counts: {bt.AssetSweepSummary(momentum_results).verdict_counts}")
    
    return results


def run_null_benchmark(tickers, lookback):
    """Run matched null benchmark (coin-flip signal) with same costs."""
    print(f"\n--- Matched null benchmark (lookback={lookback}) ---")
    
    null_results = {}
    
    for ticker, bars in tickers.items():
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)
        
        # Use the framework's noise_benchmark infrastructure to create matched null
        # This creates a coin-flip signal with identical cost structure and trading frequency
        print(f"  Testing null benchmark for {ticker}...")
        
        # Define null signals function using perturbation.random_signals
        n = len(bars)
        base_seed = SEED
        
        def null_seed(params):
            total = sum(float(v) for v in params.values())
            return base_seed + int(round(total) * 1000)
        
        def null_signals_fn(closes, **params):
            # This function never looks at closes, it's truly independent of data
            # But we need to pass the right signature for the framework
            seed = null_seed(params)
            signals = bt.perturbation.random_signals(n, seed=seed)
            return signals
        
        # Generate the null signals once
        signals = null_signals_fn(closes, lookback=lookback)
        
        # Run the null benchmark using the same stress_segments infrastructure
        null_res = bt.stress_segments(
            signals_fn=null_signals_fn,
            bars=bars,
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
        
        null_results[ticker] = {
            "segments": [s.name for s in null_res.scenarios],
            "medians": [round(s.baseline_median_log_return, 3) for s in null_res.scenarios],
            "null_medians": [round(s.noise_median_log_return, 3) for s in null_res.scenarios],
            "candidate_dispersion": round(null_res.candidate_dispersion, 3),
            "null_dispersion": round(null_res.null_dispersion, 3),
            "verdict": null_res.overall_verdict,
            "n_folds": null_res.n_folds,
        }
    
    return null_results


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== Momentum Cost Sensitivity Check ===")
    print("Testing whether short-horizon momentum edge survives conservative")
    print("transaction-cost stress on the collected real-data universe.\n")
    
    # Load the real data universe
    print("1. Manifest integrity check...")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        if status == "MISMATCH":
            print(f"  WARNING: {entry['ticker']}: checksum mismatch")
    
    print(f"  dataset: {manifest['dataset_id']} | universe: {len(manifest['universe'])} assets")
    
    print("\n2. Data preflight (research/data/preflight.py)...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)
    print("  preflight passed")
    
    # Load the real-data universe
    tickers = load_universe()
    
    # Run the cost sensitivity test
    results = run_cost_sensitivity_test(tickers)
    
    # Run matched null benchmark for comparison
    print("\n=== Matched Null Benchmark ===")
    null_results = {}
    for lookback in LOOKBACKS:
        null_results[f"lookback_{lookback}"] = run_null_benchmark(tickers, lookback)
    
    # Create artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": LOOKBACKS,
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "realistic_costs": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
            "initial_capital": REALISTIC_COSTS.initial_capital,
        },
        "cost_sensitivity_results": results,
        "null_benchmark_results": null_results,
        "interpretation": "This test determines whether the momentum edge is distinguishable",
        "from a coin-flip null when subjected to realistic transaction costs. If momentum",
        "maintains positive edge in >= 2 lookbacks under realistic costs, it demonstrates",
        "robustness to transaction costs. Otherwise, the edge is cost-sensitive or noise.",
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== Results Summary ===")
    for lookback_key, ticker_results in results.items():
        lookback = int(lookback_key.split('_')[1])
        verdict_counts = bt.AssetSweepSummary(ticker_results).verdict_counts
        edges = verdict_counts.get("EDGE", 0)
        losses = verdict_counts.get("LOSS", 0)
        no_edge = verdict_counts.get("NO_EDGE", 0)
        print(f"  lookback {lookback}: {edges} assets with positive EDGE, {losses} with LOSS,")
        print(f"                {no_edge} with NO_EDGE out of {len(ticker_results)} assets")
    
    print(f"\n=== Artifact written to: {ARTIFACT_PATH} ===")
    print("\nThis artifact can be independently verified by recomputing the")
    print("momentum cost sensitivity analysis and comparing the results.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())