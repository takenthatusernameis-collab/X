"""Cost-sensitivity test for momentum (lookback 3/5/10) with realistic costs.

This implements the R-001 task: bounded cost-sensitivity check for lookback 3/5/10
momentum under the repository's existing realistic cost model and matched null.

Primary question: Does the short-horizon momentum edge survive conservative
transaction-cost stress on the collected real-data universe?

Scope: Use existing 10-asset collected universe and existing momentum
implementation/check framework.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
LOOKBACKS = [3, 5, 10]
SEED = 42
WARMUP = 60
TRAIN = 252
TEST = 84
OVERLAP = 60
MIN_SEGMENT_BARS = 400
N_BLOCKS = 4
WINDOW = 60

# Realistic cost parameters from repository examples
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARMUP,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARMUP,
)

def momentum_signals(closes: np.ndarray, lookback: int):
    """Momentum signal: long previous N-day return, hold 1 day, daily rebalance."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return signals


def main():
    """Run cost-sensitivity test for momentum lookback variations."""
    print("=== Momentum Cost-Sensitivity Test (R-001) ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Realistic costs: commission_per_trade=${REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share=${REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents=${REALISTIC_COSTS.slippage_cents/100:.3f}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional*100:.2f}bps")
    print()

    # 1. Manifest integrity
    print("=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, f"manifest id mismatch: {manifest['dataset_id']}"
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | "
          f"universe: {len(manifest['universe'])}")

    # 2. Preflight gate
    print("\n=== 2. Data preflight ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost-sensitivity test.")
        return 1

    # 3. Load universe and run cost tests
    print("\n=== 3. Cost-sensitivity analysis ===")
    universe = manifest["universe"]
    results = {}

    # Build matched null via coin-flip: random ±1 signs
    np.random.seed(SEED)
    null_results = {}

    for lookback in LOOKBACKS:
        print(f"\n--- Lookback {lookback} ===")
        
        # Load real data for this lookback
        asset_results = {}
        null_asset_results = {}

        for ticker in universe:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()

            # Candidate: momentum signal
            cand_signals = momentum_signals(closes, lookback)
            
            # Zero cost baseline
            cfg_zero = ZERO_COSTS
            zero_result = bt.walk_forward(
                list(bars), cand_signals, train_window=TRAIN, test_window=TEST,
                warmup=WARMUP, overlap_window=OVERLAP, cfg=cfg_zero
            )
            zero_metrics = zero_result.aggregate_metrics

            # Realistic costs
            cfg_real = REALISTIC_COSTS
            real_result = bt.walk_forward(
                list(bars), cand_signals, train_window=TRAIN, test_window=TEST,
                warmup=WARMUP, overlap_window=OVERLAP, cfg=cfg_real
            )
            real_metrics = real_result.aggregate_metrics

            # Matched null: coin-flip random signals
            null_signals = [bt.Signal(date=i + 1, weight=np.random.choice([1.0, -1.0])) 
                           for i in range(len(closes))]
            null_cfg = REALISTIC_COSTS
            null_result = bt.walk_forward(
                list(bars), null_signals, train_window=TRAIN, test_window=TEST,
                warmup=WARMUP, overlap_window=OVERLAP, cfg=null_cfg
            )
            null_metrics = null_result.aggregate_metrics

            asset_results[ticker] = {
                "lookback": lookback,
                "zero_cost": {
                    "median_log_return": zero_metrics["median_log_total_return"],
                    "mean_log_return": zero_metrics["mean_log_total_return"],
                    "positive_folds": zero_metrics["fold_count_positive"],
                    "n_folds": zero_metrics["n_folds"],
                },
                "real_costs": {
                    "median_log_return": real_metrics["median_log_total_return"],
                    "mean_log_return": real_metrics["mean_log_total_return"],
                    "positive_folds": real_metrics["fold_count_positive"],
                    "n_folds": real_metrics["n_folds"],
                },
                "matched_null": {
                    "median_log_return": null_metrics["median_log_total_return"],
                    "mean_log_return": null_metrics["mean_log_total_return"],
                    "positive_folds": null_metrics["fold_count_positive"],
                    "n_folds": null_metrics["n_folds"],
                }
            }

            null_asset_results[ticker] = asset_results[ticker]

        results[lookback] = asset_results

        # Summary for this lookback
        print(f"\nLookback {lookback} Summary:")
        print("Ticker      | Zero Cost (med) | Real Cost (med) | Null (med) | Edge Surves Costs?")
        print("-" * 85)
        
        for ticker in sorted(universe):
            res = asset_results[ticker]
            zero_med = res["zero_cost"]["median_log_return"]
            real_med = res["real_costs"]["median_log_return"]
            null_med = res["matched_null"]["median_log_return"]
            survives = "YES" if real_med > null_med + 0.001 else "NO"
            print(f"{ticker:11s} | {zero_med:+.3f}        | {real_med:+.3f}         | {null_med:+.3f}     | {survives}")

        # Aggregate statistics
        all_zero = [asset_results[t]["zero_cost"]["median_log_return"] for t in universe]
        all_real = [asset_results[t]["real_costs"]["median_log_return"] for t in universe]
        all_null = [asset_results[t]["matched_null"]["median_log_return"] for t in universe]

        print(f"\n  Aggregate across {len(universe)} assets:")
        print(f"  Zero cost:   median={np.median(all_zero):+.3f}, mean={np.mean(all_zero):+.3f}")
        print(f"  Real costs:  median={np.median(all_real):+.3f}, mean={np.mean(all_real):+.3f}")
        print(f"  Matched null: median={np.median(all_null):+.3f}, mean={np.mean(all_null):+.3f}")
        
        # Determine if edge survives costs
        edge_survives = np.mean(all_real) > np.mean(all_null) + 0.001
        print(f"  Edge survives realistic costs: {edge_survives}")

    # 4. Write artifact
    print("\n=== 4. Writing artifact ===")
    artifact_path = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Simplify for artifact: keep essential information
    artifact = {
        "dataset_id": DATASET_ID,
        "lookbacks": LOOKBACKS,
        "realistic_costs": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        "results_by_lookback": {},
        "hypothesis": "Short-horizon momentum edge (lookback 3/5/10) survives realistic transaction costs",
        "decision_relevant": True,
    }

    for lookback, asset_results in results.items():
        artifact["results_by_lookback"][str(lookback)] = {
            "per_asset": {},
            "aggregate": {
                "zero_cost_median": float(np.median([asset_results[t]["zero_cost"]["median_log_return"] for t in universe])),
                "zero_cost_mean": float(np.mean([asset_results[t]["zero_cost"]["median_log_return"] for t in universe])),
                "real_cost_median": float(np.median([asset_results[t]["real_costs"]["median_log_return"] for t in universe])),
                "real_cost_mean": float(np.mean([asset_results[t]["real_costs"]["median_log_return"] for t in universe])),
                "null_median": float(np.median([asset_results[t]["matched_null"]["median_log_return"] for t in universe])),
                "null_mean": float(np.mean([asset_results[t]["matched_null"]["median_log_return"] for t in universe])),
                "edge_survives_realistic_costs": bool(np.mean([asset_results[t]["real_costs"]["median_log_return"] - asset_results[t]["matched_null"]["median_log_return"] for t in universe]) > 0.001),
            }
        }
        
        for ticker, res in asset_results.items():
            artifact["results_by_lookback"][str(lookback)]["per_asset"][ticker] = {
                "lookback": res["lookback"],
                "zero_cost_median": res["zero_cost"]["median_log_return"],
                "zero_cost_positive_folds": res["zero_cost"]["positive_folds"],
                "real_cost_median": res["real_costs"]["median_log_return"],
                "real_cost_positive_folds": res["real_costs"]["positive_folds"],
                "null_median": res["matched_null"]["median_log_return"],
                "null_positive_folds": res["matched_null"]["positive_folds"],
                "edge_survives_realistic_costs": res["real_costs"]["median_log_return"] > res["matched_null"]["median_log_return"] + 0.001,
            }

    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"  artifact written to: {artifact_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())