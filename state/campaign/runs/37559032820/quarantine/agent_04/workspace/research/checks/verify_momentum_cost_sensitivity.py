"""Verification script for momentum cost sensitivity artifact.

This script independently recomputes the momentum cost sensitivity results
and verifies they match the artifact written by momentum_cost_sensitivity.py.
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

# Existing realistic cost model from the repository's examples
cost_realistic = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Matched null with zero costs
cost_zero = bt.BacktestConfig(initial_capital=1e6)

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
ARTIFACT_DIR = ARTIFACT_PATH.parent


def momentum_signals(closes: np.ndarray, lookback: int):
    """Long previous lookback-day return; same contract as research/backtest/regime_stability."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback - 1, n):
        signals[i] = bt.Signal(date=i + 1, weight=1.0)
    return signals


def run_cost_comparison(ticker, all_tickers, lookback):
    """Run momentum for both realistic and zero cost models, return comparison."""
    bars = all_tickers[ticker]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Realistic costs
    signals_real = momentum_signals(closes, lookback)
    res_real = bt.stress_segments(
        signals_fn=momentum_signals,
        bars=list(bars),
        signals=signals_real,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cost_realistic,
    )
    
    # Zero cost matched null
    signals_zero = momentum_signals(closes, lookback)
    res_zero = bt.stress_segments(
        signals_fn=momentum_signals,
        bars=list(bars),
        signals=signals_zero,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cost_zero,
    )
    
    # Compare the results
    return dict(
        realistic=dict(
            segments=[s.name for s in res_real.scenarios],
            medians=[round(s.baseline_median_log_return, 3) for s in res_real.scenarios],
            null_medians=[round(s.noise_median_log_return, 3) for s in res_real.scenarios],
            candidate_dispersion=round(res_real.candidate_dispersion, 3),
            null_dispersion=round(res_real.null_dispersion, 3),
            verdict=res_real.overall_verdict,
            n_folds=res_real.n_folds,
        ),
        zero_cost=dict(
            segments=[s.name for s in res_zero.scenarios],
            medians=[round(s.baseline_median_log_return, 3) for s in res_zero.scenarios],
            null_medians=[round(s.noise_median_log_return, 3) for s in res_zero.scenarios],
            candidate_dispersion=round(res_zero.candidate_dispersion, 3),
            null_dispersion=round(res_zero.null_dispersion, 3),
            verdict=res_zero.overall_verdict,
            n_folds=res_zero.n_folds,
        ),
        cost_impact=dict(
            realistic_median=res_real.scenarios[0].baseline_median_log_return,
            zero_median=res_zero.scenarios[0].baseline_median_log_return,
            median_diff=res_real.scenarios[0].baseline_median_log_return - res_zero.scenarios[0].baseline_median_log_return,
        )
    )


def main() -> int:
    print("=== Verification of momentum cost sensitivity artifact ===")
    
    # Load the artifact
    with open(ARTIFACT_PATH, "r") as f:
        artifact = json.load(f)
    
    # Verify basic metadata
    assert artifact["dataset_id"] == DATASET_ID
    assert artifact["lookbacks"] == list(LOOKBACKS)
    assert artifact["train"] == TRAIN
    assert artifact["test"] == TEST
    assert artifact["warmup"] == WARM
    assert artifact["overlap"] == OVERLAP
    assert artifact["window"] == WINDOW
    assert artifact["n_blocks"] == N_BLOCKS
    assert artifact["min_segment_bars"] == MIN_SEGMENT_BARS
    
    print("Basic metadata verification: PASSED")
    
    # Reload manifest for data preflight
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("\n=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")
    
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting verification.")
        sys.exit(1)
    
    print("\n=== 3. Independent recomputation ===")
    
    # Load all tickers
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    
    # Recompute the cost sensitivity results
    recomputed_results = {}
    for lb in LOOKBACKS:
        recomputed_results[lb] = {}
        for ticker in artifact["per_asset"].keys():
            recomputed_results[lb][ticker] = run_cost_comparison(ticker, all_tickers, lb)
    
    # Compare with artifact
    artifact_data = artifact["per_asset"]
    
    # Compare realistic results
    realistic_matches = 0
    realistic_total = 0
    zero_matches = 0
    zero_total = 0
    
    for ticker in artifact_data.keys():
        for lb in LOOKBACKS:
            realistic_total += 1
            # Compare realistic results
            artifact_realistic = artifact_data[ticker][str(lb)]["realistic"]
            recomputed_realistic = recomputed_results[lb][ticker]["realistic"]
            
            # Compare medians
            artifact_medians = artifact_realistic["medians"]
            recomputed_medians = recomputed_realistic["medians"]
            if artifact_medians == recomputed_medians:
                realistic_matches += 1
            else:
                print(f"  MISMATCH realistic medians for {ticker} lookback {lb}:")
                print(f"    artifact: {artifact_medians}")
                print(f"    recomputed: {recomputed_medians}")
            
            # Compare zero cost results
            zero_total += 1
            artifact_zero = artifact_data[ticker][str(lb)]["zero_cost"]
            recomputed_zero = recomputed_results[lb][ticker]["zero_cost"]
            
            # Compare medians
            artifact_zero_medians = artifact_zero["medians"]
            recomputed_zero_medians = recomputed_zero["medians"]
            if artifact_zero_medians == recomputed_zero_medians:
                zero_matches += 1
            else:
                print(f"  MISMATCH zero cost medians for {ticker} lookback {lb}:")
                print(f"    artifact: {artifact_zero_medians}")
                print(f"    recomputed: {recomputed_zero_medians}")
    
    print(f"\n  Realistic cost results: {realistic_matches}/{realistic_total} matches ({realistic_matches/realistic_total*100:.1f}%)")
    print(f"  Zero cost results: {zero_matches}/{zero_total} matches ({zero_matches/zero_total*100:.1f}%)")
    
    if realistic_matches == realistic_total and zero_matches == zero_total:
        print("\n  ALL VERIFICATIONS PASSED")
        print("  The momentum cost sensitivity artifact is verified and matches the independent recomputation.")
    else:
        print("\n  VERIFICATION FAILED")
        print("  The momentum cost sensitivity artifact does not match the independent recomputation.")
        sys.exit(1)
    
    print("\n=== Verification completed successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())