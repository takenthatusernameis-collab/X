"""Cost robustness check for lookback 3/5/10 momentum.

Background: The momentum lookback sweep (activation 37361000967) shows that
lookback 3/5/10 produce REGIME_STABLE positive edges in 6/10 assets, with
robustness verdicts across multiple lookbacks. The edge appears to be short-
horizon and mechanical. Cost robustness remains decision-relevant: we need to
verify that the edge survives realistic transaction costs.

Objective: Run a bounded cost-sensitivity check for lookback 3/5/10 momentum
under the repository's existing realistic cost model and matched null.

Method: Use the existing 10-asset collected universe and existing momentum
implementation. Long the previous N-day return (lookback=3,5,10), hold 1 day,
daily rebalance. Run walk-forward IS/OOS with realistic costs and with a
coin-flip null (matched across the same segments, same costs).

Realistic cost model (from ma_crossover.py examples):
- commission_per_trade: 2.0
- commission_per_share: 0.003
- slippage_cents: 2.0
- slippage_proportional: 0.0005

Walk-forward parameters: train=252d/test=84d/warmup=60d/overlap=60d,
min 400 bars/segment, 4 contiguous volatility blocks as in existing checks.

This is a bounded check, not a scope expansion: only lookback 3/5/10 on the
collected universe, with cost comparison only.
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
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cost_robustness_check_results.json"


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Cost Robustness Check for Lookback 3/5/10 Momentum ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Universe: {len(manifest['universe'])} assets")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Walk-forward: train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d")
    print(f"Min segment bars: {MIN_SEGMENT_BARS}")
    print()

    # 1. Manifest integrity
    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print()

    # 2. Data preflight
    print("=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost robustness check.")
        sys.exit(1)
    print("Preflight passed.")
    print()

    # 3. Cost-robustness check for each lookback
    print("=== 3. Cost-robustness check (realistic costs vs matched null) ===")
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    results = {}
    for lb in LOOKBACKS:
        print(f"\n--- Lookback {lb} days ---")
        
        # Load universe data
        closes_data = {}
        for ticker, bars in tickers_univ.items():
            closes_data[ticker] = bars.closes_array()
        
        # Run regime stability with realistic costs
        print(f"  Running regime stability with realistic costs...")
        summary_real = bt.stress_segments_across_tickers(
            tickers=tickers_univ,
            signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lb),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", lb),),
            param_grid=[{"lookback": lb}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=REALISTIC_COSTS,
        )
        
        # Run coin-flip null with realistic costs
        # For noise_benchmark, we need to convert the BarSequence objects to a single list
        # of Bar objects, not a list of BarSequence objects
        from research.backtest.data import BarSequence
        if all(isinstance(bars, BarSequence) for bars in tickers_univ.values()):
            # All are BarSequence, convert each to list and combine
            all_bars_as_list = []
            for bars in tickers_univ.values():
                all_bars_as_list.extend(list(bars))
        else:
            # Already in correct format
            all_bars_as_list = list(tickers_univ.values())
        
        print(f"  Running coin-flip null with realistic costs...")
        null_result = bt.noise_benchmark(
            bars=all_bars_as_list,
            param_grid=[{"lookback": lb}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            seed=SEED,
            cfg=REALISTIC_COSTS,
        )
        
        # Calculate aggregate metrics from per-asset results
        # Since we don't have aggregate_metrics from the universe summary, we need to compute them
        all_folds = []
        
        for ticker, stress_result in summary_real.assets.items():
            # Each stress result contains per-segment metrics
            # We need to aggregate across segments and assets
            for scenario in stress_result.scenarios:
                all_folds.append(scenario.baseline_median_log_return)
        
        # Store results
        results[f"lookback_{lb}"] = {
            "realistic_cost_median_log_return": float(np.median(all_folds) if all_folds else 0.0),
            "realistic_cost_mean_log_return": float(np.mean(all_folds) if all_folds else 0.0),
            "realistic_cost_n_folds": len(all_folds),
            "null_median_log_return": float(null_result.baseline_median_log_return),
            "null_vs_realistic_cost_diff": float(np.median(all_folds) - null_result.baseline_median_log_return) if all_folds else 0.0,
            "null_vs_realistic_cost_significance": abs(np.median(all_folds) - null_result.baseline_median_log_return) > 0.05 if all_folds else False,  # 5% tolerance
        }
        
        # Print summary for this lookback
        print(f"  Realistic cost: median log return = {results[f'lookback_{lb}']['realistic_cost_median_log_return']:+.3f}")
        print(f"  Null (coin-flip): median log return = {results[f'lookback_{lb}']['null_median_log_return']:+.3f}")
        print(f"  Difference: {results[f'lookback_{lb}']['null_vs_realistic_cost_diff']:+.3f}")
        print(f"  Significant edge (vs null): {results[f'lookback_{lb}']['null_vs_realistic_cost_significance']}")

    # 4. Summary and determinism check
    print("\n=== 4. Summary ===")
    print("Cost-robustness check completed for lookbacks 3/5/10.")
    print("Key findings:")
    
    edge_survives_costs = []
    for lb in LOOKBACKS:
        significant = results[f"lookback_{lb}"]["null_vs_realistic_cost_significance"]
        median_ret = results[f"lookback_{lb}"]["realistic_cost_median_log_return"]
        edge_survives_costs.append(significant and median_ret > 0)
        print(f"  Lookback {lb}: significant edge vs null = {significant}, median log return = {median_ret:+.3f}")
    
    all_survive = all(edge_survives_costs)
    print(f"\nAll lookbacks survive realistic costs (significant + positive): {all_survive}")

    # 5. Write artifact
    print("\n=== 5. Writing artifact ===")
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_cost_model=dict(
            commission_per_trade=REALISTIC_COSTS.commission_per_trade,
            commission_per_share=REALISTIC_COSTS.commission_per_share,
            slippage_cents=REALISTIC_COSTS.slippage_cents,
            slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        ),
        results=results,
        summary=dict(
            all_lookbacks_survive_realistic_costs=all_survive,
            edge_survives_costs=edge_survives_costs,
            cost_robustness_conclusion="ROBUST" if all_survive else "SENSITIVE" if any(edge_survives_costs) else "NO_EDGE",
        ),
    )
    
    # Convert artifact to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)  # Convert numpy bool to Python bool
        elif isinstance(obj, (int, np.integer)):
            return int(obj)
        elif isinstance(obj, (float, np.floating)):
            return float(obj)
        elif isinstance(obj, (str, type(None))):
            return obj
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    artifact_serializable = make_serializable(artifact)
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact_serializable, f, indent=2, sort_keys=True)
    print(f"Artifact written to: {ARTIFACT_PATH}")

    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Cost robustness check for lookback 3/5/10 momentum: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())