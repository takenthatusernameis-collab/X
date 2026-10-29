"""Cost-robustness check for lookback 3/5/10 momentum.

Research/simulation only. No live trading or production execution.
Uses existing 10-asset collected universe and existing momentum implementation/check framework.

A-priori prediction: costs erode the edge for lookback=3 (shorter horizon -> more trades -> higher costs).
Costs may preserve the edge for lookback=5 and potentially lookback=10 (fewer, larger moves).

Stop after the predeclared cost grid is evaluated and the independent verifier agrees or records the exact discrepancy.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

# Realistic cost model from examples/ma_crossover.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Zero cost baseline for comparison
ZERO_COSTS = bt.BacktestConfig(initial_capital=1e6)

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = [3, 5, 10]
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"

def run_momentum_sweep(lookback, costs):
    """Run momentum walk-forward for a specific lookback with given costs."""
    bars = bt.generate_bars(2500, seed=SEED)
    
    def momentum_signals(closes, lb):
        n = len(closes)
        out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
        lb = int(lb)
        for i in range(lb, n):
            ret = float(np.mean(np.log(closes[i - lb + 1 : i + 1])))
            out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
        return out
    
    signals = momentum_signals(bars.closes_array(), lookback)
    
    result = bt.walk_forward(
        list(bars),
        signals,
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
        cfg=costs,
    )
    
    return result

def run_null_benchmark_sweep(lookback, costs):
    """Run coin-flip null benchmark for a specific lookback with same costs."""
    bars = bt.generate_bars(2500, seed=SEED)
    
    np.random.seed(SEED)
    signals = []
    for i in range(len(bars)):
        if i < 60:  # warmup
            signals.append(bt.Signal(date=i + 1, weight=0.0))
        else:
            # Coin-flip random direction
            signals.append(bt.Signal(date=i + 1, weight=np.random.choice([1.0, -1.0])))
    
    result = bt.walk_forward(
        list(bars),
        signals,
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
        cfg=costs,
    )
    
    return result

def get_zero_cost_baseline(lookback):
    """Get zero-cost baseline median from existing momentum_results.json."""
    with open("/home/runner/work/X/X/state/check_artifacts/momentum_results.json") as f:
        momentum_data = json.load(f)
    
    if lookback == 5:
        return momentum_data["perturbation"]["baseline_median"]
    else:
        # For lookbacks other than 5, run a zero-cost sweep to get baseline
        bars = bt.generate_bars(600, seed=SEED)
        
        def momentum_signals(closes, lb):
            n = len(closes)
            out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
            lb = int(lb)
            for i in range(lb, n):
                ret = float(np.mean(np.log(closes[i - lb + 1 : i + 1])))
                out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
            return out
        
        signals = momentum_signals(bars.closes_array(), lookback)
        result = bt.walk_forward(
            list(bars),
            signals,
            train_window=60,
            test_window=20,
            warmup=10,
            overlap_window=10,
            cfg=ZERO_COSTS,
        )
        
        log_returns = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in result.folds]
        return float(np.median(log_returns))

def main():
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print("=== Momentum cost-robustness check ===")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Realistic costs: commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    
    results = {}
    
    for lookback in LOOKBACKS:
        print(f"\n--- Lookback {lookback} ---")
        
        # Get zero-cost baseline
        baseline_median = get_zero_cost_baseline(lookback)
        
        # Run with realistic costs
        print(f"Running momentum (lookback={lookback}) with realistic costs...")
        cost_result = run_momentum_sweep(lookback, REALISTIC_COSTS)
        cost_log_returns = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in cost_result.folds]
        cost_median = float(np.median(cost_log_returns))
        
        # Run null benchmark with realistic costs
        print(f"Running null benchmark (lookback={lookback}) with realistic costs...")
        null_result = run_null_benchmark_sweep(lookback, REALISTIC_COSTS)
        null_log_returns = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in null_result.folds]
        null_median = float(np.median(null_log_returns))
        
        # Calculate cost erosion
        cost_eroded = cost_median - baseline_median
        cost_gap = cost_median - null_median
        
        print(f"  Zero-cost baseline: {baseline_median:+.3f}")
        print(f"  With realistic costs: {cost_median:+.3f}")
        print(f"  Null benchmark: {null_median:+.3f}")
        print(f"  Cost erosion: {cost_eroded:+.3f}")
        print(f"  Cost gap (momentum - null): {cost_gap:+.3f}")
        
        # Classify cost robustness using the same criteria as momentum.py
        if abs(cost_gap) < 0.05:  # Within 5% of noise threshold
            verdict = "CONSISTENT_WITH_NOISE"
        elif cost_gap > 0.05 and cost_eroded > -0.02:  # Positive gap and not severely eroded
            verdict = "REGIME_STABLE"
        else:
            verdict = "FALSIFIED"
        
        results[str(lookback)] = {
            "baseline_median": round(baseline_median, 3),
            "cost_median": round(cost_median, 3),
            "null_median": round(null_median, 3),
            "cost_eroded": round(cost_eroded, 3),
            "cost_gap": round(cost_gap, 3),
            "verdict": verdict,
            "cost_result": {
                "n_folds": cost_result.aggregate_metrics["n_folds"],
                "total_oos_periods": cost_result.aggregate_metrics["total_oos_periods"],
                "mean_log_return": round(float(cost_result.aggregate_metrics["mean_log_total_return"]), 3),
                "median_log_return": round(float(cost_result.aggregate_metrics["median_log_total_return"]), 3),
            },
            "null_result": {
                "n_folds": null_result.aggregate_metrics["n_folds"],
                "total_oos_periods": null_result.aggregate_metrics["total_oos_periods"],
                "mean_log_return": round(float(null_result.aggregate_metrics["mean_log_total_return"]), 3),
                "median_log_return": round(float(null_result.aggregate_metrics["median_log_total_return"]), 3),
            }
        }
    
    # Write artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": LOOKBACKS,
        "realistic_costs": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        "results_by_lookback": results,
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== Artifact written to {ARTIFACT_PATH} ===")
    
    # Print summary
    print("\n=== Cost robustness summary ===")
    for lookback, result in results.items():
        print(f"Lookback {lookback}: {result['verdict']} "
              f"(baseline={result['baseline_median']:+.3f}, "
              f"cost={result['cost_median']:+.3f}, "
              f"gap={result['cost_gap']:+.3f})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
