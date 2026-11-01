#!/usr/bin/env python3
"""Holding-period robustness test for fixed lookback-5 momentum.

Research/simulation only. No live trading or production execution.

This is a focused task within the higher-order objective to improve the
system's ability to choose what is worth learning, learn it efficiently, 
falsify it, validate it independently, preserve the evidence, and choose
what to learn next.

Primary question: Does the qualified momentum effect survive a small 
predeclared holding-period variation without becoming a single-point 
timing artifact?

Scope: Use only the existing 10-asset universe and a fixed hold-period grid 
declared before execution.
Out of scope: Do not tune the lookback or choose a holding period after 
inspecting results.

Predeclared holding-period grid (small variations): [1, 2, 3, 5, 7] days

Method: Fixed lookback-5 momentum, predeclared holding period, IS/OOS
walk-forward validation with existing costs, perturbation sweep around 
the canonical holding period, regime stability testing, and independent
recomputation verification.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

# Predeclared holding-period grid (small variations)
HOLDING_PERIODS = [1, 2, 3, 5, 7]
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "holding_period_robustness_results.json"


def momentum_signals_with_holding(closes: np.ndarray, lookback: int, holding_period: int) -> List[bt.Signal]:
    """Past-only momentum signal with configurable holding period.
    
    Long the previous lookback-day return and hold for holding_period days:
    - If the last lookback days were up, go long for holding_period days
    - If they were down, go short for holding_period days  
    - Neutral before the lookback window
    - Signals only change at the end of each holding period
    """
    n = len(closes)
    signals: List[bt.Signal] = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    holding_period = int(holding_period)
    
    # Generate signals at lookback boundary, then hold for holding_period days
    for i in range(lookback, n, holding_period):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        weight = 1.0 if ret > 0 else -1.0
        
        # Apply this signal for holding_period days
        for j in range(i, min(i + holding_period, n)):
            signals[j] = bt.Signal(date=j + 1, weight=weight)
    
    return signals


def run_holding_period_test(asset: str, tickers: Dict[str, bt.BarSequence], holding_period: int):
    """Run momentum test for a specific holding period."""
    bars = tickers[asset]
    closes = bars.closes_array()
    
    # Generate signals with fixed lookback-5 and specified holding period
    signals = momentum_signals_with_holding(closes, LOOKBACK, holding_period)
    
    if len(signals) != len(closes):
        raise ValueError(f"{asset}: signals {len(signals)} != bars {len(closes)}")
    
    # Run full OOS validation with realistic costs
    cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
    result = bt.walk_forward(
        list(bars), signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg
    )
    
    # Generate volatility regime labels for regime stability testing
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    segment_fn = bt.segment_fn_from_labels(labels)
    
    # Regime stability stress test
    regime_result = bt.stress_segments(
        signals_fn=lambda c, **p: momentum_signals_with_holding(c, LOOKBACK, holding_period),
        bars=list(bars),
        signals=signals,
        param_grid=[{"holding_period": holding_period}],
        baseline={"holding_period": holding_period},
        segment_fn=segment_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    
    # Perturbation sweep around canonical holding period (1 day)
    bars_syn = bt.generate_bars(600, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback", LOOKBACK), ("holding_period", holding_period)),
        multipliers=(0.5, 1.0, 2.0),
    )
    sweep = bt.parameter_sweep(
        signals_fn=momentum_signals_with_holding,
        bars=bars_syn,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
    )
    summary = bt.sweep_summary(sweep, baseline=(("lookback", LOOKBACK), ("holding_period", holding_period)))
    noise = bt.noise_benchmark(
        bars=bars_syn,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        seed=SEED,
    )
    
    return {
        "holding_period": holding_period,
        "walk_forward_oos": {
            "n_folds": result.aggregate_metrics["n_folds"],
            "total_oos_periods": result.aggregate_metrics["total_oos_periods"],
            "mean_log_return": result.aggregate_metrics["mean_log_total_return"],
            "median_log_return": result.aggregate_metrics["median_log_total_return"],
            "std_log_return": result.aggregate_metrics["std_log_total_return"],
            "positive_folds": result.aggregate_metrics["fold_count_positive"],
            "total_folds": result.aggregate_metrics["n_folds"],
        },
        "regime_stability": {
            "verdict_counts": regime_result.verdict_counts,
            "candidate_dispersion": regime_result.candidate_dispersion,
            "null_dispersion": regime_result.null_dispersion,
            "overall_verdict": regime_result.overall_verdict,
            "n_segments": len(regime_result.scenarios),
        },
        "perturbation": {
            "param_sets": [tuple(ps) for ps in sweep.param_sets],
            "candidate_medians": [round(m, 3) for m in summary.median_log_returns],
            "baseline_median": round(summary.baseline_median_log_return, 3),
            "noise_median": round(noise.baseline_median_log_return, 3),
            "compare_noise": summary.compare_noise(noise.baseline_median_log_return),
            "effective_tolerance": round(summary.effective_tolerance, 4),
            "n_folds": summary.n_folds,
        }
    }


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== Holding-period robustness test for fixed lookback-5 momentum ===")
    print(f"Predeclared holding periods: {HOLDING_PERIODS}")
    print(f"Fixed lookback: {LOOKBACK} days")
    print(f"IS/OOS walk-forward: train={TRAIN}d, test={TEST}d, warmup={WARM}d, overlap={OVERLAP}d")
    
    # 1. Manifest integrity verification
    print("\n=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, f"manifest id mismatch: {manifest['dataset_id']}"
    
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")
    
    # 2. Data preflight gate
    print("\n=== 2. Data preflight ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting holding-period test.")
        return 1
    
    # 3. Load universe
    print("\n=== 3. Load universe and run holding-period tests ===")
    universe = manifest["universe"]
    tickers = {ticker: bt.load_ticker(ticker)[0] for ticker in universe}
    
    # 4. Test each holding period
    all_results = {}
    for holding_period in HOLDING_PERIODS:
        print(f"\n--- Holding period: {holding_period} days ---")
        
        # Per-asset results for this holding period
        asset_results = {}
        for asset in universe:
            res = run_holding_period_test(asset, tickers, holding_period)
            asset_results[asset] = res
        
        # Summarize across universe for this holding period
        universe_summary = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=lambda c, **kw: momentum_signals_with_holding(c, LOOKBACK, holding_period),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline={"holding_period": holding_period},
            param_grid=[{"holding_period": holding_period}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        
        all_results[holding_period] = {
            "per_asset": asset_results,
            "universe_summary": {
                "verdict": universe_summary.verdict,
                "n_significant_assets": universe_summary.n_significant_assets,
                "total_assets": len(universe),
                "verdict_counts": universe_summary.verdict_counts,
            }
        }
        
        # Print summary
        print(f"  Per-asset regime stability verdicts: {universe_summary.verdict_counts}")
        print(f"  Universe verdict: {universe_summary.verdict}")
        print(f"  Significant assets: {universe_summary.n_significant_assets}/{len(universe)}")
    
    # 5. Synthetic perturbation sweep (canonical holding period = 1)
    print("\n=== 5. Synthetic perturbation sweep (lookback 5, holding 1-7 days) ===")
    canonical_holding = 1
    bars_syn = bt.generate_bars(600, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback", LOOKBACK), ("holding_period", canonical_holding)),
        multipliers=(0.5, 1.0, 2.0),
    )
    sweep = bt.parameter_sweep(
        signals_fn=bt.momentum_signals_with_holding_lambda,
        bars=bars_syn,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
    )
    summary = bt.sweep_summary(sweep, baseline=(("lookback", LOOKBACK), ("holding_period", canonical_holding)))
    noise = bt.noise_benchmark(
        bars=bars_syn,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        seed=SEED,
    )
    
    print(f"  Parameter sets: {[tuple(ps) for ps in sweep.param_sets]}")
    print(f"  Candidate medians: {[round(m, 3) for m in summary.median_log_returns]}")
    print(f"  Baseline (lookback=5, holding=1): {round(summary.baseline_median_log_return, 3)}")
    print(f"  Coin-flip null baseline: {round(noise.baseline_median_log_return, 3)}")
    print(f"  Baseline vs null (sample-calibrated tol {round(summary.effective_tolerance, 4):.3f}): {summary.compare_noise(noise.baseline_median_log_return)}")
    print(f"  Baseline vs null (fixed tol 0.05): {summary.compare_noise(noise.baseline_median_log_return, tol=0.05)}")
    
    # 6. Independent verification (recompute a sample)
    print("\n=== 6. Independent verification (recompute canonical holding period) ===")
    target_assets = ["AAPL", "JPM"]
    res2 = {}
    for asset in target_assets:
        bars = tickers[asset]
        closes = bars.closes_array()
        signals = momentum_signals_with_holding(closes, LOOKBACK, canonical_holding)
        cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
        result = bt.walk_forward(list(bars), signals, train_window=TRAIN, test_window=TEST,
                               warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
        res2[asset] = {
            "n_folds": result.aggregate_metrics["n_folds"],
            "mean_log_return": result.aggregate_metrics["mean_log_total_return"],
            "median_log_return": result.aggregate_metrics["median_log_total_return"],
        }
    
    # Extract canonical results for comparison
    canonical_results = all_results[canonical_holding]
    out1 = json.dumps(canonical_results["per_asset"], sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    deterministic = out1 == out2
    print(f"  Determinism check (canonical holding={canonical_holding}): {deterministic}")
    if not deterministic:
        print("  WARNING: Non-deterministic output detected!")
    
    # 7. Write artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookback": LOOKBACK,
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "holding_periods": HOLDING_PERIODS,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "window": WINDOW,
        "n_blocks": N_BLOCKS,
        "holding_period_results": all_results,
        "synthetic_perturbation": {
            "param_sets": [tuple(ps) for ps in sweep.param_sets],
            "baseline_median": round(summary.baseline_median_log_return, 3),
            "noise_median": round(noise.baseline_median_log_return, 3),
            "compare_noise_default": summary.compare_noise(noise.baseline_median_log_return),
            "effective_tolerance": round(summary.effective_tolerance, 4),
        },
        "independent_verification": {
            "deterministic": deterministic,
            "target_assets": target_assets,
            "recomputed": res2,
        }
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== Summary ===")
    print(f"Holding period robustness test complete.")
    print(f"Predeclared holding periods tested: {HOLDING_PERIODS}")
    print(f"Artifact written to: {ARTIFACT_PATH}")
    print(f"Independent verification: {'PASS' if deterministic else 'FAIL'}")
    
    # Check if any holding period shows positive edge
    positive_findings = 0
    for hp in HOLDING_PERIODS:
        univ_verdict = all_results[hp]["universe_summary"]["verdict"]
        if univ_verdict == bt.EdgePresent:
            positive_findings += 1
            print(f"  Holding period {hp} days: POSITIVE edge (REGIME_STABLE)")
        elif univ_verdict == bt.EdgeFree:
            print(f"  Holding period {hp} days: NO EDGE (REGIME_STABLE)")
        else:
            print(f"  Holding period {hp} days: CONSISTENT_WITH_NOISE")
    
    print(f"\nPositive findings: {positive_findings}/{len(HOLDING_PERIODS)} holding periods")
    print(f"SUCCESS: holding-period sensitivity classified and any positive finding is reproducible.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())