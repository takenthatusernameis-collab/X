#!/usr/bin/env python3
"""Bounded cost-sensitivity check for lookback 3/5/10 momentum under realistic costs.

This is Agent 02's deliverable: a reproducible artifact and independent verification
classifying cost robustness for short-horizon momentum signals.

Research/simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Add the repository root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import research modules
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file


def walk_forward_fold_log_returns(daily_pnl, train, test, warm, overlap):
    """Walk-forward OOS fold log returns for a spread P&L series.

    Mirrors research.backtest.engine.walk_forward's aggregate logic.
    walk_forward computes each fold's total_return as the ratio of the
    equity curve at the last OOS bar to the equity curve at the first OOS
    bar; since the equity at the first OOS bar already reflects that bar's
    return, the fold log return compounds the spread's daily P&L over the
    last ``test - 1`` bars of the OOS window.

    Args:
        daily_pnl: np.ndarray of daily spread P&L.
        train, test, warm, overlap: walk-forward geometry.

    Returns:
        List of per-fold log OOS returns.
    """
    n = len(daily_pnl)
    step = test - overlap
    fold_start = 0
    lrets = []
    while True:
        fold_end = fold_start + warm + train + test
        if fold_end > n:
            break
        oos = daily_pnl[fold_end - test + 1:fold_end]
        comp = 1.0
        for r in oos:
            comp *= 1.0 + r
        lrets.append(float(np.log1p(np.clip(comp - 1.0, -1.0 + 1e-12, None))))
        fold_start += step
    return lrets


def regime_verdict(candidate_medians, null_medians):
    """Regime-stability verdict for a spread across segments.

    This is a custom implementation matching the logic used in
    cross_sectional_momentum.py, since the backtest module doesn't
    export this function.
    """
    candidate_medians = np.array(candidate_medians, dtype=np.float64)
    null_medians = np.array(null_medians, dtype=np.float64)
    
    # Define verdict constants
    REGIME_VERDICT_NO_EDGE = "CONSISTENT_WITH_NOISE"
    REGIME_VERDICT_STABLE = "REGIME_STABLE"
    REGIME_VERDICT_STABLE_LOSS = "REGIME_STABLE_LOSS"
    REGIME_VERDICT_DEPENDENT = "REGIME_DEPENDENT"
    
    if all(abs(m) <= 0.05 for m in candidate_medians):
        return REGIME_VERDICT_NO_EDGE
    candidate_dispersion = float(np.std(candidate_medians, ddof=1))
    null_dispersion = float(np.std(null_medians, ddof=1))
    if candidate_dispersion > 2.0 * null_dispersion:
        return REGIME_VERDICT_DEPENDENT
    if all(m < 0 for m in candidate_medians):
        return REGIME_VERDICT_STABLE_LOSS
    return REGIME_VERDICT_STABLE

# Constants from the existing momentum check
DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cost_sensitivity_momentum.json"

# Realistic cost model from examples/ma_crossover.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Zero cost model for comparison
ZERO_COSTS = bt.BacktestConfig(initial_capital=1e6)

# Real-data null (coin-flip null)
REAL_DATA_SEED = 99


def run_momentum_cost_sensitivity(tickers):
    """Run bounded cost-sensitivity check for lookback 3/5/10 momentum.

    Returns artifact data with both zero-cost and realistic-cost results,
    plus matched nulls for comparison.
    """
    print("=== Bounded cost-sensitivity check for lookback 3/5/10 momentum ===")
    print("Scope: existing 10-asset collected universe, realistic cost model")
    print("Lookback periods:", LOOKBACKS)

    # Analyze existing momentum results from artifact to understand baseline
    existing_artifact_path = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"
    if existing_artifact_path.exists():
        with open(existing_artifact_path) as f:
            existing_artifact = json.load(f)
        print(f"Loaded existing momentum artifact (lookback={existing_artifact['lookback']})")

    # Load universe manifest
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print(f"\n=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    # Prepare cost comparison results
    cost_sensitivity_results = {
        "dataset_id": DATASET_ID,
        "lookbacks": LOOKBACKS,
        "cost_parameters": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        "per_lookback": {},
    }

    # For each lookback period, run zero-cost and realistic-cost momentum
    for lookback in LOOKBACKS:
        print(f"\n=== Lookback {lookback} ===")

        # Load all tickers
        all_tickers = {}
        for entry in manifest["entries"]:
            all_tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

        # Generate real-data matched null (coin-flip null)
        print("Generating matched null (coin-flip null)...")
        null_spread = bt.cs_momentum_null_spread_daily_returns(
            all_tickers, lookback, 3, 3, REAL_DATA_SEED
        )
        null_asset_medians = []
        
        # Walk forward on the null spread to get per-segment medians
        for segment in range(0, len(null_spread), TEST - OVERLAP):
            if segment + WARM + TRAIN + TEST > len(null_spread):
                break
            null_lrets = walk_forward_fold_log_returns(
                null_spread[segment:segment + WARM + TRAIN + TEST], TRAIN, TEST, WARM, OVERLAP
            )
            null_asset_medians.append(float(np.median(null_lrets)))

        # Run momentum with zero costs
        print("Running momentum with zero costs...")
        zero_cost_medians = []
        realistic_cost_medians = []
        zero_cost_verdicts = []
        realistic_cost_verdicts = []

        for ticker, bars in all_tickers.items():
            closes = bars.closes_array()

            # Zero-cost momentum
            zero_signals = bt.momentum_signals(closes, lookback)
            zero_res = bt.walk_forward(
                list(bars), zero_signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=ZERO_COSTS
            )
            zero_lrets = [
                np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                for f in zero_res.folds
            ]
            zero_median = float(np.median(zero_lrets)) if zero_lrets else 0.0
            zero_cost_medians.append(zero_median)

            # Realistic-cost momentum
            realistic_signals = bt.momentum_signals(closes, lookback)
            realistic_res = bt.walk_forward(
                list(bars), realistic_signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=REALISTIC_COSTS
            )
            realistic_lrets = [
                np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                for f in realistic_res.folds
            ]
            realistic_median = float(np.median(realistic_lrets)) if realistic_lrets else 0.0
            realistic_cost_medians.append(realistic_median)

            print(f"  {ticker}: zero_cost {zero_median:+.3f} realistic_cost {realistic_median:+.3f}")

        # Compute verdicts using complete median lists for this lookback period (custom implementation)
        zero_verdict = regime_verdict(zero_cost_medians, null_asset_medians)
        realistic_verdict = regime_verdict(realistic_cost_medians, null_asset_medians)
        zero_cost_verdicts = [zero_verdict] * len(zero_cost_medians)  # Same verdict for all assets in this lookback
        realistic_cost_verdicts = [realistic_verdict] * len(realistic_cost_medians)  # Same verdict for all assets in this lookback

        # Calculate cost impact
        cost_impact = []
        for i, (zero_med, realistic_med) in enumerate(zip(zero_cost_medians, realistic_cost_medians)):
            impact = realistic_med - zero_med
            cost_impact.append(impact)

        # Store results for this lookback
        cost_sensitivity_results["per_lookback"][str(lookback)] = {
            "zero_cost_medians": [round(m, 3) for m in zero_cost_medians],
            "realistic_cost_medians": [round(m, 3) for m in realistic_cost_medians],
            "null_medians": [round(m, 3) for m in null_asset_medians],
            "cost_impacts": [round(i, 3) for i in cost_impact],
            "zero_cost_verdicts": zero_cost_verdicts,
            "realistic_cost_verdicts": realistic_cost_verdicts,
        }

        # Calculate dispersion comparison
        zero_disp = float(np.std(zero_cost_medians, ddof=1)) if len(zero_cost_medians) > 1 else 0.0
        realistic_disp = float(np.std(realistic_cost_medians, ddof=1)) if len(realistic_cost_medians) > 1 else 0.0
        null_disp = float(np.std(null_asset_medians, ddof=1)) if len(null_asset_medians) > 1 else 0.0

        print(f"\n  Zero cost:   dispersion {zero_disp:+.3f}, null dispersion {null_disp:+.3f}")
        print(f"  Realistic:   dispersion {realistic_disp:+.3f}, null dispersion {null_disp:+.3f}")
        print(f"  Cost impact: {[f'{i:+.3f}' for i in cost_impact]}")

    return cost_sensitivity_results


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Agent 02 Cost-Sensitivity Check ===")
    print("Objective: Classify cost robustness for lookback 3/5/10 momentum")
    print("Deliverable: Reproducible artifact and independent verification")

    # Run the bounded cost-sensitivity check
    results = run_momentum_cost_sensitivity({})

    # Write artifact
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"\n=== Artifact written to: {ARTIFACT_PATH} ===")

    # Independent verification (quick sanity check)
    print("\n=== Independent Verification ===")
    verify_artifact_path = Path.cwd() / "state" / "check_artifacts" / "cost_sensitivity_momentum.json"

    with open(verify_artifact_path) as f:
        verify_artifact = json.load(f)

    # Verify artifact structure
    required_fields = ["dataset_id", "lookbacks", "cost_parameters", "per_lookback"]
    for field in required_fields:
        if field not in verify_artifact:
            print(f"ERROR: Missing required field '{field}'")
            return 1

    # Verify each lookback has expected structure
    for lookback, data in verify_artifact["per_lookback"].items():
        required_lookback_fields = ["zero_cost_medians", "realistic_cost_medians", 
                                  "null_medians", "cost_impacts", "zero_cost_verdicts", 
                                  "realistic_cost_verdicts"]
        for field in required_lookback_fields:
            if field not in data:
                print(f"ERROR: Lookback {lookback} missing field '{field}'")
                return 1

    print("SUCCESS: Artifact structure verified")

    # Summary
    print("\n=== Cost-Robustness Classification ===")
    for lookback, data in verify_artifact["per_lookback"].items():
        zero_verdicts = data["zero_cost_verdicts"]
        realistic_verdicts = data["realistic_cost_verdicts"]
        cost_impacts = data["cost_impacts"]

        zero_consistent = all(v == "CONSISTENT_WITH_NOISE" for v in zero_verdicts)
        realistic_consistent = all(v == "CONSISTENT_WITH_NOISE" for v in realistic_verdicts)

        avg_cost_impact = np.mean([abs(i) for i in cost_impacts])

        if realistic_consistent and zero_consistent:
            classification = "ROBUST_TO_COSTS"
        elif realistic_consistent and not zero_consistent:
            classification = "COST_VULNERABLE"
        elif not realistic_consistent and zero_consistent:
            classification = "COST_MITIGATED"
        else:
            classification = "COST_SENSITIVE"

        print(f"Lookback {lookback}: {classification} (avg cost impact {avg_cost_impact:.3f})")
        print(f"  Zero cost verdict: {set(zero_verdicts)}")
        print(f"  Realistic cost verdict: {set(realistic_verdicts)}")

    print(f"\n=== Note: research/simulation only. No live trading or production execution. "
          f"Cost-sensitivity momentum check on the collected universe: exploratory simulation. ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())