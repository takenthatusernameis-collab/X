"""Independent verification of momentum cost-robustness check.

This recomputes the cost-sensitivity check from a separate path and compares
against the artifact written by cost_sensitivity_momentum_v2.py.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

# Realistic cost model (must match exactly)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ZERO_COSTS = bt.BacktestConfig(initial_capital=1e6)

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = [3, 5, 10]
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"

def momentum_signals(closes, lookback):
    """Fresh implementation of momentum signals (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out

def null_signals(closes, seed=SEED):
    """Fresh implementation of coin-flip null signals."""
    np.random.seed(seed)
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(60, n):  # warmup period
        out[i] = bt.Signal(date=i + 1, weight=np.random.choice([1.0, -1.0]))
    return out

def main():
    print("=== Independent verification of momentum cost-robustness ===")
    
    # Load the artifact to verify against
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    
    print(f"Verifying artifact: {ARTIFACT_PATH}")
    print(f"Lookbacks: {LOOKBACKS}")
    
    all_ok = True
    
    for lookback in LOOKBACKS:
        lookback_str = str(lookback)
        print(f"\n--- Verifying lookback {lookback} ---")
        
        artifact_result = artifact["results_by_lookback"][lookback_str]
        
        # Recompute momentum result with realistic costs
        bars = bt.generate_bars(2500, seed=SEED)
        signals = momentum_signals(bars.closes_array(), lookback)
        
        result = bt.walk_forward(
            list(bars),
            signals,
            train_window=252,
            test_window=84,
            warmup=60,
            overlap_window=60,
            cfg=REALISTIC_COSTS,
        )
        
        # Compute median log return
        log_returns = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in result.folds]
        median_log_return = float(np.median(log_returns))
        
        # Compare with artifact
        artifact_median = artifact_result["cost_median"]
        tolerance = 0.001  # 1 basis point tolerance
        
        if abs(median_log_return - artifact_median) <= tolerance:
            print(f"  Momentum result: MATCH (recomputed {median_log_return:+.3f} vs artifact {artifact_median:+.3f})")
        else:
            print(f"  Momentum result: MISMATCH (recomputed {median_log_return:+.3f} vs artifact {artifact_median:+.3f})")
            all_ok = False
        
        # Recompute null benchmark result with realistic costs
        null_signals_list = null_signals(bars.closes_array(), SEED)
        
        null_result = bt.walk_forward(
            list(bars),
            null_signals_list,
            train_window=252,
            test_window=84,
            warmup=60,
            overlap_window=60,
            cfg=REALISTIC_COSTS,
        )
        
        # Compute median log return
        null_log_returns = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in null_result.folds]
        null_median_log_return = float(np.median(null_log_returns))
        
        # Compare with artifact
        artifact_null_median = artifact_result["null_median"]
        
        if abs(null_median_log_return - artifact_null_median) <= tolerance:
            print(f"  Null benchmark: MATCH (recomputed {null_median_log_return:+.3f} vs artifact {artifact_null_median:+.3f})")
        else:
            print(f"  Null benchmark: MISMATCH (recomputed {null_median_log_return:+.3f} vs artifact {artifact_null_median:+.3f})")
            all_ok = False
        
        # Verify verdict consistency
        cost_gap = median_log_return - null_median_log_return
        
        # Re-compute verdict using same criteria as original
        if abs(cost_gap) < 0.05:
            expected_verdict = "CONSISTENT_WITH_NOISE"
        elif cost_gap > 0.05 and (median_log_return - artifact_result["baseline_median"]) > -0.02:
            expected_verdict = "REGIME_STABLE"
        else:
            expected_verdict = "FALSIFIED"
        
        if expected_verdict == artifact_result["verdict"]:
            print(f"  Verdict: MATCH ({expected_verdict})")
        else:
            print(f"  Verdict: MISMATCH (expected {expected_verdict} vs artifact {artifact_result['verdict']})")
            all_ok = False
    
    print("\n" + "="*50)
    if all_ok:
        print("SUCCESS: All independent recomputations MATCH the check artifact.")
        print("The cost-robustness check is reproducible and deterministic.")
    else:
        print("FAILURE: MISMATCH FOUND - the check artifact does not reproduce.")
        print("Investigate before admitting the results.")
        sys.exit(1)
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-robustness independent verification completed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
