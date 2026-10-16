"""Cost-sensitivity check for momentum signals with realistic transaction costs.

Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the
repository's existing realistic cost model and matched null.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATA_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42

# Existing realistic cost model from repository (commission_per_share=5e-06, slippage_proportional=0.0005)
COST_MODEL = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_share=5e-06,
    slippage_proportional=0.0005,
    warmup_periods=0,
    periods_per_year=252,
)

# Existing null cost model with matching parameters for fair comparison
NULL_COST_MODEL = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_share=5e-06,
    slippage_proportional=0.0005,
    warmup_periods=0,
    periods_per_year=252,
)

LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400


def run_cost_robustness_check():
    """Run cost robustness check for momentum lookbacks 3, 5, 10."""
    
    print("=== Cost Robustness Check for Momentum (lookbacks 3/5/10) ===")
    print(f"Cost model: commission_per_share=5e-06, slippage_proportional=0.0005")
    print(f"Dataset: {DATA_ID}")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Walk-forward: train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d")
    print(f"Min segment bars: {MIN_SEGMENT_BARS}, Volatility blocks: {N_BLOCKS}")
    print()

    # Load real data manifest
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATA_ID, f"Dataset mismatch: {manifest['dataset_id']}"
    
    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")
    print()

    # Load universe data
    print("=== 2. Data preflight ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost robustness check.")
        sys.exit(1)
    print("PREFLIGHT PASSED")
    print()

    # Load all tickers
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars

    # Cost robustness check per asset
    print("=== 3. Cost robustness check per asset ===")
    
    results = {}
    for ticker in tickers:
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)
        
        asset_results = {}
        for lookback in LOOKBACKS:
            # Generate momentum signals
            signals = bt.momentum_signals(closes, lookback=lookback)
            if len(signals) != len(closes):
                raise ValueError(f"{ticker} {lookback}: signals {len(signals)} != bars {len(closes)}")
            
            # Run backtest with realistic cost model
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
            )
            
            # Generate matched null with same cost model
            null_signals = bt.random_signals(closes, weight_range=(0.0, 0.0))
            null_res = bt.stress_segments(
                signals_fn=bt.random_signals,
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
            )
            
            # Compute cost robustness metric (median log return difference)
            cost_robustness = res.baseline_median_log_return - null_res.baseline_median_log_return
            
            asset_results[str(lookback)] = {
                "candidate_median_log_return": round(res.baseline_median_log_return, 3),
                "null_median_log_return": round(null_res.baseline_median_log_return, 3),
                "cost_robustness": round(cost_robustness, 3),
                "candidate_dispersion": round(res.candidate_dispersion, 3),
                "null_dispersion": round(null_res.candidate_dispersion, 3),
                "segments": [s.name for s in res.scenarios],
                "verdict": res.overall_verdict,
                "null_verdict": null_res.overall_verdict,
                "n_folds": res.n_folds,
                "edge_significant": abs(cost_robustness) > max(0.05, 2 * null_res.candidate_dispersion),
            }
        
        results[ticker] = asset_results
        
        # Print summary for this asset
        print(f"{ticker}:")
        for lookback in LOOKBACKS:
            r = asset_results[str(lookback)]
            sig = "*" if r["edge_significant"] else " "
            print(f"  lookback {lookback}: candidate={r['candidate_median_log_return']:+.3f}, "
                  f"null={r['null_median_log_return']:+.3f}, "
                  f"cost_robust={r['cost_robustness']:+.3f}{sig} "
                  f"verdict={r['verdict']} null={r['null_verdict']}")
    
    print()
    
    # Summary statistics
    print("=== 4. Summary statistics ===")
    total_edges = 0
    significant_edges = 0
    for ticker in results:
        for lookback in LOOKBACKS:
            if results[ticker][str(lookback)]["edge_significant"]:
                total_edges += 1
                significant_edges += 1
    
    print(f"Total positive cost-robustness edges: {significant_edges}/{total_edges}")
    print(f"Cost-robustness significance threshold: max(0.05, 2 * null dispersion)")
    
    # Count robust vs non-robust
    robust_assets = 0
    sensitive_assets = 0
    no_edge_assets = 0
    
    for ticker in results:
        edges = [results[ticker][str(lookback)]["edge_significant"] for lookback in LOOKBACKS]
        if any(edges):
            if all(edges):
                robust_assets += 1
            else:
                sensitive_assets += 1
        else:
            no_edge_assets += 1
    
    print(f"Robust assets (>=2 lookbacks positive): {robust_assets}")
    print(f"Sensitive assets (1 lookback positive): {sensitive_assets}")
    print(f"No-edge assets: {no_edge_assets}")
    
    return results


def main() -> int:
    ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"
    
    results = run_cost_robustness_check()
    
    # Write artifact
    artifact = {
        "dataset_id": DATA_ID,
        "seed": SEED,
        "lookbacks": LOOKBACKS,
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "cost_model": {
            "commission_per_share": COST_MODEL.commission_per_share,
            "slippage_proportional": COST_MODEL.slippage_proportional,
        },
        "per_asset": results,
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Cost robustness check completed: momentum lookback 3/5/10 vs matched null.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())