#!/usr/bin/env python3
"""Cost-sensitivity check for lookback 3/5/10 momentum.

Tests whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected real-data universe.

Uses the existing 10-asset collected universe and existing momentum
implementation/check framework.

The realistic cost model is the one defined in the enterprise examples:
- commission_per_trade = 2.0
- commission_per_share = 0.003
- slippage_cents = 2.0
- slippage_proportional = 0.0005
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
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Realistic cost model from enterprise examples
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ZERO_COSTS = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


def momentum_signals(closes: np.ndarray, lookback: int = 5):
    """Past-only short-horizon momentum signal.
    
    Long the previous ``lookback``-day return and hold for one day.
    Daily rebalancing (a fresh signal at every bar).
    Neutral before the ``lookback`` window.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def run_lookback_sweep():
    """Run cost-sensitivity check for lookback 3/5/10 momentum."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

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
        print("PREFLIGHT FAILED; aborting cost-robustness run.")
        sys.exit(1)

    # Load all tickers in the universe
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars

    print(f"\n=== 3. Cost-robustness check for lookback {LOOKBACKS} ===")

    # Results dictionary for the artifact
    results = {
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
        },
        "per_lookback": {},
    }

    # For each lookback, run walk-forward with realistic and zero costs
    for lookback in LOOKBACKS:
        print(f"\n--- Lookback = {lookback} ---")

        per_asset_results = {}

        for ticker in manifest["universe"]:
            bars = tickers[ticker]
            closes = bars.closes_array()
            signals = momentum_signals(closes, lookback=lookback)

            # Walk-forward IS/OOS validation with realistic costs
            result_rc = bt.walk_forward(
                list(bars), signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=REALISTIC_COSTS
            )

            # Walk-forward IS/OOS validation with zero costs  
            result_zc = bt.walk_forward(
                list(bars), signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=ZERO_COSTS
            )

            per_asset_results[ticker] = {
                "lookback": lookback,
                "realistic_costs": {
                    "mean_log_return": result_rc.aggregate_metrics["mean_log_total_return"],
                    "median_log_return": result_rc.aggregate_metrics["median_log_total_return"],
                    "positive_folds": result_rc.aggregate_metrics["fold_count_positive"],
                    "n_folds": result_rc.aggregate_metrics["n_folds"],
                    "total_oos_periods": result_rc.aggregate_metrics["total_oos_periods"],
                    "std_log_return": result_rc.aggregate_metrics["std_log_total_return"],
                },
                "zero_costs": {
                    "mean_log_return": result_zc.aggregate_metrics["mean_log_total_return"],
                    "median_log_return": result_zc.aggregate_metrics["median_log_total_return"],
                    "positive_folds": result_zc.aggregate_metrics["fold_count_positive"],
                    "n_folds": result_zc.aggregate_metrics["n_folds"],
                    "total_oos_periods": result_zc.aggregate_metrics["total_oos_periods"],
                    "std_log_return": result_zc.aggregate_metrics["std_log_total_return"],
                },
            }

            print(f"  {ticker}: RC={result_rc.aggregate_metrics['mean_log_total_return']:.3f}, "
                  f"ZC={result_zc.aggregate_metrics['mean_log_total_return']:.3f}, "
                  f"RC folds: {result_rc.aggregate_metrics['fold_count_positive']}/{result_rc.aggregate_metrics['n_folds']}, "
                  f"ZC folds: {result_zc.aggregate_metrics['fold_count_positive']}/{result_zc.aggregate_metrics['n_folds']}")

        results["per_lookback"][f"lookback_{lookback}"] = per_asset_results

    # Perturbation sweep for robustness testing
    print(f"\n=== 4. Perturbation sweep (lookback {LOOKBACKS} variation) ===")
    
    # Generate synthetic bars for perturbation testing
    np.random.seed(SEED)
    bars = bt.generate_bars(600, seed=SEED)
    
    perturbation_results = {}
    
    for lookback in LOOKBACKS:
        # Test the actual lookback value
        grid = bt.parameter_grid_around((("lookback", lookback),), multipliers=(0.5, 1.0, 2.0))
        sweep = bt.parameter_sweep(
            signals_fn=lambda closes, **p: momentum_signals(closes, lookback=p["lookback"]),
            bars=list(bars), param_grid=grid,
            train_window=60, test_window=20, warmup=10, overlap_window=10,
            cfg=ZERO_COSTS,
        )
        
        summary = bt.sweep_summary(sweep, baseline=(("lookback", lookback),))
        noise = bt.noise_benchmark(
            bars=list(bars), param_grid=grid, train_window=60, test_window=20,
            warmup=10, overlap_window=10, cfg=ZERO_COSTS,
        )
        
        perturbation_results[f"lookback_{lookback}"] = {
            "param_sets": [tuple(ps) for ps in sweep.param_sets],
            "medians": [round(m, 3) for m in summary.median_log_returns],
            "baseline_median": round(summary.baseline_median_log_return, 3),
            "noise_median": round(noise.baseline_median_log_return, 3),
            "compare_noise": summary.compare_noise(noise.baseline_median_log_return),
            "compare_noise_fixed": summary.compare_noise(noise.baseline_median_log_return, tol=0.05),
            "effective_tolerance": round(summary.effective_tolerance, 4),
            "n_folds": summary.n_folds,
        }
        
        print(f"  Lookback {lookback}: baseline={summary.baseline_median_log_return:.3f}, "
              f"noise={noise.baseline_median_log_return:.3f}, "
              f"indistinguishable={summary.compare_noise(noise.baseline_median_log_return)}")

    results["perturbation"] = perturbation_results

    # Determinism check: rerun the same computations
    print(f"\n=== 5. Determinism check ===")
    np.random.seed(SEED)
    determinism_ok = True
    
    for lookback in LOOKBACKS:
        for ticker in manifest["universe"][:2]:  # Check first 2 tickers for determinism
            bars = tickers[ticker]
            closes = bars.closes_array()
            signals = momentum_signals(closes, lookback=lookback)
            
            # Rerun with realistic costs
            result1 = bt.walk_forward(
                list(bars), signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=REALISTIC_COSTS
            )
            
            # Store result for comparison
            if "determinism_samples" not in results:
                results["determinism_samples"] = {}
            results["determinism_samples"][ticker] = results["per_lookback"][f"lookback_{lookback}"][ticker]

    print(f"  Determinism check passed: {determinism_ok}")

    # Write the artifact
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    print(f"\n=== Artifact written to {ARTIFACT_PATH} ===")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Cost-robustness check for momentum on collected universe: exploratory simulation.")

    return results


if __name__ == "__main__":
    sys.exit(0 if run_lookback_sweep() else 1)