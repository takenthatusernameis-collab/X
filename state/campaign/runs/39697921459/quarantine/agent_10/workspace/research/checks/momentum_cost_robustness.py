"""Cost-sensitivity check for lookback 3/5/10 momentum.

This script implements the bounded cost-sensitivity check required by
R-001: Global Agent 10 (campaign slot 10) to evaluate whether the short-horizon
momentum edge survives conservative transaction-cost stress on the collected
10-asset real-data universe.

Research methodology:
- Test momentum signals with lookback 3, 5, 10 days under realistic costs
- Test the same signals under zero costs (matched null)
- Use walk-forward IS/OOS validation with established parameters
- Compare performance to assess cost robustness
- Preserve exact methodology for independent verification

Cost model (realistic):
- commission_per_trade: 2.0
- commission_per_share: 0.003
- slippage_cents: 2.0
- slippage_proportional: 0.0005

This matches the cfg1 configuration in examples/ma_crossover.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

# Constants from the momentum check (R-001 contract)
DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Realistic cost model (from examples/ma_crossover.py cfg1)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Zero cost model (matched null)
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
)

def momentum_signals(closes, lookback):
    """Momentum signal implementation (independent code path for verification)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out

def segment_median(ticker: str, s: int, e: int, lookback: int) -> Tuple[float, int]:
    """Independent verification implementation from verify_momentum.py."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = bt.BacktestConfig(warmup_periods=WARM)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)

def run_cost_sensitivity():
    """Main cost-sensitivity analysis."""
    print("=== Cost-sensitivity check for lookback 3/5/10 momentum ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Universe size: 10 assets")
    print(f"Walk-forward: train={TRAIN}d / test={TEST}d / warmup={WARM}d / overlap={OVERLAP}d")
    print()

    # Load the manifest to get the 10-asset universe
    from research.data.preflight import load_manifest
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID
    universe = manifest["universe"]
    print(f"Assets in universe: {universe}")
    print()

    # Test results storage
    results = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "lookback_cost_sensitivity": {},
        "independent_verification": {}
    }

    # For each lookback, run realistic cost and zero cost comparisons
    for lookback in [3, 5, 10]:
        print(f"--- Lookback = {lookback} ---")
        lookback_results = {}

        for cost_model, cost_name in [(REALISTIC_COSTS, "realistic"), (ZERO_COSTS, "zero")]:
            print(f"\nCost model: {cost_name.upper()}")

            asset_results = {}
            all_segment_medians = []

            # Test across all assets in the universe
            for ticker in universe:
                # Load ticker data
                bars, dates = bt.load_ticker(ticker)
                closes = bars.closes_array()

                # Generate momentum signals
                signals = momentum_signals(closes, lookback)

                # Run walk-forward validation
                res = bt.walk_forward(
                    list(bars), signals,
                    train_window=TRAIN, test_window=TEST,
                    warmup=WARM, overlap_window=OVERLAP,
                    cfg=cost_model)

                # Collect segment medians
                segment_medians = []
                for fold in res.folds:
                    log_ret = np.log1p(np.clip(fold.metrics["total_return"], -1.0 + 1e-12, None))
                    segment_medians.append(log_ret)

                asset_segment_medians = list(np.array(segment_medians))
                asset_results[ticker] = {
                    "n_folds": res.n_folds,
                    "segment_medians": [round(m, 4) for m in asset_segment_medians],
                    "mean_median": round(float(np.mean(asset_segment_medians)), 4),
                    "median_median": round(float(np.median(asset_segment_medians)), 4),
                    "std_median": round(float(np.std(asset_segment_medians)), 4),
                }
                all_segment_medians.extend(asset_segment_medians)

            # Calculate universe-level statistics
            if all_segment_medians:
                mean_median = float(np.mean(all_segment_medians))
                median_median = float(np.median(all_segment_medians))
                std_median = float(np.std(all_segment_medians))
                positive_pct = (sum(1 for m in all_segment_medians if m > 0) / len(all_segment_medians)) * 100

                lookback_results[cost_name] = {
                    "universe_statistics": {
                        "n_assets": len(universe),
                        "total_folds": sum(asset_results[t]["n_folds"] for t in universe),
                        "mean_segment_median": round(mean_median, 4),
                        "median_segment_median": round(median_median, 4),
                        "std_segment_median": round(std_median, 4),
                        "positive_fold_percentage": round(positive_pct, 1),
                    },
                    "per_asset": asset_results
                }

                print(f"  Universe stats: mean={mean_median:+.4f}, median={median_median:+.4f}, std={std_median:.4f}, pos%={positive_pct:.1f}%")

        results["lookback_cost_sensitivity"][f"lookback_{lookback}"] = lookback_results

        # Cost robustness analysis
        realistic = lookback_results["realistic"]["universe_statistics"]["mean_segment_median"]
        zero_cost = lookback_results["zero"]["universe_statistics"]["mean_segment_median"]
        cost_impact = realistic - zero_cost

        print(f"\n  Cost impact: realistic ({realistic:+.4f}) - zero ({zero_cost:+.4f}) = {cost_impact:+.4f}")
        if abs(cost_impact) > 0.001:
            print(f"  -> MATERIAL COST SENSITIVITY (difference > 0.1% per fold)")
        else:
            print(f"  -> INSIGNIFICANT COST SENSITIVITY (difference < 0.1% per fold)")

        print()

    # Save results for verification
    output_path = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Results saved to: {output_path}")
    print("\n=== Cost-sensitivity check completed ===")
    return results

if __name__ == "__main__":
    run_cost_sensitivity()