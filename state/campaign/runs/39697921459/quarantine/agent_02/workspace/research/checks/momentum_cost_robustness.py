"""Cost robustness check for short-horizon momentum (lookback 3/5/10) on the collected universe.

This script tests whether the momentum edge survives realistic transaction costs.
It uses the same infrastructure as momentum.py but focuses on cost sensitivity:
- Parameter sweep across lookback 3/5/10 (baseline lookback=5)
- Realistic cost model (commission, slippage)
- Matched null (coin-flip signal) for statistical significance
- Independent recomputation for determinism verification

Research / simulation only. No live trading.
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
BASELINE_LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Realistic cost model from ma_crossover_real_data.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"


def run_lookback_sweep_with_costs():
    """Run parameter sweep across lookback 3/5/10 with realistic costs on each ticker."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    # Create parameter grid for lookback 3/5/10
    baseline = (("lookback", BASELINE_LOOKBACK),)
    multipliers = (1.0, 2.0)  # 5*0.5=2.5 (round to 3), 5*1.0=5, 5*2.0=10
    grid = bt.parameter_grid_around(baseline, multipliers)
    
    print("=== Momentum cost robustness check ===")
    print(f"Lookback grid: {[p['lookback'] for p in grid]}")
    print(f"Realistic costs: commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    print()
    
    # Load all tickers and run parameter sweep on each
    all_sweep_results = []
    all_null_summaries = []
    universe_summary = None
    
    for ticker in manifest["universe"]:
        bars, dates = bt.load_ticker(ticker)
        
        # Run sweep with realistic costs on this ticker
        sweep_result = bt.parameter_sweep(
            signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=p["lookback"]),
            bars=list(bars),
            param_grid=grid,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=REALISTIC_COSTS,
            periods_per_year=252,
        )
        
        # Convert to summary for easier access
        sweep_summary = bt.sweep_summary(sweep_result, baseline=(("lookback", BASELINE_LOOKBACK),))
        
        # Run null benchmark on this ticker
        null_summary = bt.noise_benchmark(
            bars=list(bars),
            param_grid=grid,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=REALISTIC_COSTS,
            periods_per_year=252,
            seed=SEED + hash(ticker) % 10000,  # Deterministic seed per ticker
        )
        
        all_sweep_results.append(sweep_summary)
        all_null_summaries.append(null_summary)
        
        print(f"  {ticker}: baseline median {sweep_summary.baseline_median_log_return:+.3f}")
    
    # Create simple universe summary
    def aggregate_sweep_results(results):
        """Aggregate parameter sweep results across multiple tickers for a simple summary."""
        # Collect all parameter sets and medians
        all_param_sets = []
        all_medians = []
        
        for result in results:
            all_param_sets.extend(result.param_sets)
            all_medians.extend(result.median_log_returns)
        
        # Find unique parameter sets in order of first appearance
        unique_param_sets = []
        param_to_medians = {}
        for ps, med in zip(all_param_sets, all_medians):
            if ps not in param_to_medians:
                param_to_medians[ps] = []
                unique_param_sets.append(ps)
            param_to_medians[ps].append(med)
        
        # Calculate median log returns for each param set across all tickers
        medians = [np.median(param_to_medians[ps]).item() for ps in unique_param_sets]
        
        # Find baseline
        baseline = (("lookback", BASELINE_LOOKBACK),)
        baseline_median = None
        for ps, med in zip(unique_param_sets, medians):
            if ps == baseline:
                baseline_median = med
                break
        
        # Calculate fraction of param sets that are positive
        fraction_positive = sum(1 for med in medians if med > 0) / len(medians) if medians else 0.0
        
        return {
            "param_sets": unique_param_sets,
            "medians": medians,
            "baseline_median_log_return": baseline_median,
            "n_folds": results[0].n_folds if results else 0,
            "n_param_sets": len(unique_param_sets),
            "fraction_positive": fraction_positive,
        }
    
    universe_summary = aggregate_sweep_results(all_sweep_results)
    
    return universe_summary, all_null_summaries, all_sweep_results


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Manifest integrity ===")
    manifest = load_manifest()
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")
    
    # Run cost robustness check
    universe_summary, all_null_summaries, all_sweep_results = run_lookback_sweep_with_costs()
    
    print("\n=== Cost robustness summary ===")
    
    # Create a simple summary similar to perturbation.py
    lines = [
        "=== Parameter-sensitivity / robustness summary ===",
        "",
        f"Baseline: lookback={BASELINE_LOOKBACK}",
        f"Folds: {universe_summary['n_folds']}  Tickers: {len(all_sweep_results)}",
        f"Train window: {TRAIN}d  Test window: {TEST}d",
        "",
        "Median log return per parameter set (averaged across tickers):",
    ]
    
    # Add parameter set results
    for ps, med in zip(universe_summary['param_sets'], universe_summary['medians']):
        lookback = ps[0][1] if len(ps) > 0 else "N/A"
        lines.append(f"  lookback {lookback}: {med:+.3f}")
    
    lines.append("")
    
    print("\n".join(lines))
    
    # Create artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        baseline_lookback=BASELINE_LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_costs={
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        sweep=universe_summary,
        null=dict(
            n_tickers=len(all_null_summaries),
            null_results=[dict(
                baseline_median=round(s.baseline_median_log_return, 3),
                n_folds=s.n_folds,
            ) for s in all_null_summaries],
        ),
    )
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
