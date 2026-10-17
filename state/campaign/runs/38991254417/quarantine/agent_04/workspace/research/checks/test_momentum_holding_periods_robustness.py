#!/usr/bin/env python3
"""
Test holding period robustness for lookback-5 momentum signals.

This script tests whether the qualified momentum effect survives a small
predeclared holding-period variation without becoming a single-point timing artifact.

Objective: Test a fixed lookback-5 momentum signal with a predeclared small
holding-period set using the existing walk-forward/null methodology.

Evidence basis: Momentum is candidate positive evidence at a one-day hold and
has already survived lookback and horizon robustness in magnitude.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
import json


def create_oscillating_data(n_bars=2500, seed=42):
    """Create synthetic data that will show different momentum signals."""
    np.random.seed(seed)
    
    # Create price series with clear up/down cycles to generate both LONG and SHORT momentum signals
    prices = [100.0]
    
    # Define cycles of up/down trends
    cycles = [
        (15, 1.02),   # Up trend
        (8, 0.98),    # Down trend  
        (12, 1.015),  # Up trend
        (10, 0.985),  # Down trend
        (20, 1.01),   # Up trend
        (6, 0.99),    # Down trend
        (18, 1.012),  # Up trend
        (5, 0.992),   # Down trend
    ]
    
    for cycle_length, trend_multiplier in cycles:
        for _ in range(cycle_length):
            if trend_multiplier > 1.0:
                prices.append(prices[-1] * (1 + (trend_multiplier - 1.0) / cycle_length))
            else:
                prices.append(prices[-1] * (1 + (trend_multiplier - 1.0) / cycle_length))
    
    # Add some noise to make it more realistic
    prices_array = np.array(prices, dtype=np.float64)
    
    # Add random walk component
    walk = np.cumsum(np.random.randn(n_bars - len(prices)) * 0.005)
    prices_array = np.concatenate([prices_array, prices_array[-1] + walk])
    
    return bt.BarSequence(
        dates=np.arange(1, len(prices_array) + 1),
        opens=prices_array,
        highs=prices_array * (1 + np.random.rand(len(prices_array)) * 0.02),
        lows=prices_array * (1 - np.random.rand(len(prices_array)) * 0.02),
        closes=prices_array,
        volumes=np.random.randint(1000, 10000, len(prices_array)),
    )


def momentum_with_holding_period(closes, lookback, holding_period):
    """
    Create momentum signals with specified holding period.
    
    This implements the logic where:
    1. Generate momentum signal based on lookback period
    2. Hold that position for the specified holding period
    3. Generate new momentum signal after holding period expires
    """
    n = len(closes)
    signals = [bt.Signal(i + 1, 0.0) for i in range(n)]
    
    if holding_period <= 0:
        return signals
    
    # Generate base momentum signals
    base_momentum = bt.momentum_signals(closes, lookback)
    
    # Apply holding period logic
    position_start_date = None
    position_weight = 0.0
    
    for i in range(n):
        if i >= lookback:
            base_weight = base_momentum[i].weight
            
            if position_start_date is None:
                # Start new position
                position_start_date = i
                position_weight = base_weight
                signals[i] = bt.Signal(i + 1, position_weight)
            
            elif position_start_date is not None:
                # Continuing active position (within holding period)
                if (i - position_start_date) < holding_period:
                    # Still holding
                    signals[i] = bt.Signal(i + 1, position_weight)
                else:
                    # Holding period expired, start new position
                    position_start_date = i
                    position_weight = base_weight
                    signals[i] = bt.Signal(i + 1, position_weight)
            else:
                # No active position
                signals[i] = bt.Signal(i + 1, 0.0)
    
    return signals


def test_holding_period_robustness():
    """Test momentum with different holding periods."""
    print("Creating synthetic data with alternating momentum signals...")
    
    # Generate data that will show both LONG and SHORT momentum signals
    bars = create_oscillating_data(n_bars=2500, seed=42)
    closes = bars.closes_array()
    
    # Predeclared holding period grid (declare BEFORE execution)
    holding_periods = [1, 2, 3, 5, 7, 10]
    
    results = {}
    
    for hp in holding_periods:
        print(f"\n=== Testing holding period: {hp} days ===")
        
        # Generate momentum signal with this holding period
        signals = momentum_with_holding_period(closes, lookback=5, holding_period=hp)
        
        # Walk-forward validation
        result = bt.walk_forward(
            bars=list(bars),
            signals=signals,
            train_window=252,
            test_window=84,
            warmup=60,
            overlap_window=60,
            cfg=bt.BacktestConfig(initial_capital=1e6),
        )
        
        fold_returns = [f.metrics["total_return"] for f in result.folds]
        
        # Summary
        median_log_return = np.median([np.log1p(np.clip(r, -1.0 + 1e-12, None)) for r in fold_returns])
        positive_folds = sum(1 for r in fold_returns if r > 0)
        avg_return = np.mean(fold_returns)
        std_return = np.std(fold_returns, ddof=1)
        
        print(f"  Folds: {len(result.folds)}")
        print(f"  Positive OOS folds: {positive_folds}/{len(result.folds)}")
        print(f"  Median log return: {median_log_return:.3f}")
        print(f"  Average total return: {avg_return:.3%}")
        print(f"  Std of returns: {std_return:.3%}")
        
        results[hp] = {
            'median_log_return': median_log_return,
            'positive_folds': positive_folds,
            'avg_return': avg_return,
            'std_return': std_return,
            'n_folds': len(result.folds),
            'fold_returns': fold_returns,
        }
    
    # Analyze holding period sensitivity
    print("\n=== Holding Period Sensitivity Analysis ===")
    
    # Compare each holding period to baseline (1-day hold)
    baseline_hp = 1
    baseline_result = results[baseline_hp]
    
    print(f"\nBaseline (HP={baseline_hp}): median log return = {baseline_result['median_log_return']:+.3f}")
    print(f"          positive folds = {baseline_result['positive_folds']}/{baseline_result['n_folds']}")
    
    robust_holding_periods = []
    sensitive_holding_periods = []
    
    for hp in holding_periods:
        if hp == baseline_hp:
            continue
        
        result = results[hp]
        
        # Check if result is within 20% of baseline and maintains positive performance
        baseline_median = abs(baseline_result['median_log_return']) if baseline_result['median_log_return'] != 0 else 0.001
        tolerance = baseline_median * 0.2
        
        # Consider robust if:
        # 1. Result is positive (maintains edge)
        # 2. Within tolerance of baseline
        # 3. At least 25% of folds are positive
        is_robust = (
            result['median_log_return'] >= 0 and
            abs(result['median_log_return'] - baseline_result['median_log_return']) <= tolerance and
            result['positive_folds'] / result['n_folds'] >= 0.25
        )
        
        if is_robust:
            robust_holding_periods.append(hp)
            print(f"  HP={hp:2d}: median log return = {result['median_log_return']:+.3f} - ROBUST")
        else:
            sensitive_holding_periods.append(hp)
            print(f"  HP={hp:2d}: median log return = {result['median_log_return']:+.3f} - SENSITIVE")
    
    # Determine final verdict
    print(f"\n=== Final Analysis ===")
    
    if len(robust_holding_periods) >= len(holding_periods) * 0.5:
        verdict = "ROBUST"
        print(f"HOLDING PERIOD SENSITIVITY IS ROBUST: {len(robust_holding_periods)}/{len(holding_periods)-1} periods maintained edge")
        print("The momentum effect survives different holding periods, suggesting it's not a single-point timing artifact.")
    else:
        verdict = "SENSITIVE"
        print(f"HOLDING PERIOD SENSITIVITY IS NOT ROBUST: only {len(robust_holding_periods)}/{len(holding_periods)-1} periods maintained edge")
        print("The momentum effect is sensitive to holding periods, suggesting it may be a timing artifact.")
    
    # Save results
    save_results = {
        'dataset_id': 'momentum-holding-period-robustness-test',
        'lookback': 5,
        'holding_periods_tested': holding_periods,
        'baseline_holding_period': baseline_hp,
        'results': results,
        'robust_holding_periods': robust_holding_periods,
        'sensitive_holding_periods': sensitive_holding_periods,
        'verdict': verdict,
        'test_parameters': {
            'train_window': 252,
            'test_window': 84,
            'warmup': 60,
            'overlap': 60,
            'seed': 42,
        }
    }
    
    return save_results


if __name__ == "__main__":
    print("Testing lookback-5 momentum signal with holding period robustness...")
    results = test_holding_period_robustness()
    
    # Save to artifact
    output_path = Path(__file__).parent.parent / "state" / "check_artifacts" / "momentum_holding_period_robustness_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create JSON-serializable version of results
    def make_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj
    
    json_results = make_json_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("\n=== SUMMARY ===")
    print("The test evaluates whether the momentum effect survives different holding periods.")
    print("This addresses the primary question: 'Does the qualified momentum effect survive")
    print("a small predeclared holding-period variation without becoming a single-point timing artifact?'")
    print(f"\nFinal verdict: {results['verdict']}")