#!/usr/bin/env python3
"""
Lookback-5 momentum signal with holding period grid test.

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

# Add the repository root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
import json


def create_holding_period_momentum_signal(closes, lookback, holding_period):
    """
    Create a momentum signal with specified holding period.
    
    For each bar i >= lookback, if the lookback period was up, generate a long
    signal that will be held for 'holding_period' days. If the lookback period
    was down, generate a short signal to be held for 'holding_period' days.
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    
    # First pass: determine position signals (0 = neutral, 1 = long, -1 = short)
    position_signals = [0] * n
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        position_signals[i] = 1 if ret > 0 else -1
    
    # Second pass: apply holding periods
    # For each position, hold for the specified number of days
    active_positions = {}  # date -> remaining holding days
    for i in range(n):
        # Check if we're starting a new position
        if position_signals[i] != 0:
            active_positions[i] = holding_period
            signals[i] = bt.Signal(date=i + 1, weight=float(position_signals[i]))
        # Check if we're continuing an existing position
        elif i in active_positions and active_positions[i] > 0:
            # Continue the position
            remaining_pos = list(active_positions.keys())[0] if active_positions else None
            if remaining_pos is not None:
                signals[i] = bt.Signal(date=i + 1, weight=float(position_signals[remaining_pos]))
                active_positions[i] = active_positions[remaining_pos] - 1
            # Remove expired positions
            if active_positions and active_positions[i] <= 0:
                del active_positions[i]
        # Otherwise stay neutral
        else:
            signals[i] = bt.Signal(date=i + 1, weight=0.0)
    
    return signals


def momentum_with_holding_periods(closes, lookback, holding_period):
    """Simple momentum signal with holding period."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    
    if holding_period <= 0:
        return out
    
    # Generate base momentum signals
    base_signals = bt.momentum_signals(closes, lookback=lookback)
    
    # For each holding period, maintain position
    active_positions = {}  # date -> (weight, remaining_days)
    for i in range(n):
        # Start new position if base signal is non-zero and no position active
        if base_signals[i].weight != 0 and i not in active_positions:
            active_positions[i] = (base_signals[i].weight, holding_period)
        
        # Check active positions
        expired_positions = []
        for start_date, (weight, remaining) in active_positions.items():
            days_held = i - start_date
            if days_held < remaining:
                # Still holding - update remaining days
                active_positions[start_date] = (weight, remaining - days_held)
                out[i] = bt.Signal(date=i + 1, weight=weight)
            else:
                # Position expired
                expired_positions.append(start_date)
        
        # Remove expired positions
        for start_date in expired_positions:
            del active_positions[start_date]
        
        # If we have no active positions but a base signal suggests one, start it
        if not active_positions and base_signals[i].weight != 0:
            active_positions[i] = (base_signals[i].weight, holding_period)
            out[i] = bt.Signal(date=i + 1, weight=base_signals[i].weight)
        
        # If we have active positions and haven't set signal for this bar yet
        if active_positions and out[i].weight == 0.0:
            # Get the earliest position
            start_date, (weight, remaining) = next(iter(active_positions.items()))
            out[i] = bt.Signal(date=i + 1, weight=weight)
    
    return out


def test_holding_period_grid():
    """Test momentum with different holding periods."""
    # Generate synthetic data
    np.random.seed(42)
    bars = bt.generate_bars(
        n_bars=2500,
        regimes=[
            bt.Regime(drift_annual=0.03, vol_annual=0.18),
            bt.Regime(drift_annual=-0.01, vol_annual=0.35),
        ],
        p_transition=0.008,
        start_price=100.0,
        seed=42,
    )
    
    closes = bars.closes_array()
    
    # Predeclared holding period grid (declare BEFORE execution)
    holding_periods = [1, 2, 3, 5, 7, 10]
    
    results = {}
    
    for hp in holding_periods:
        print(f"\n=== Testing holding period: {hp} days ===")
        
        # Generate momentum signal with this holding period
        signals = momentum_with_holding_periods(closes, lookback=5, holding_period=hp)
        
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
        
        print(f"  Folds: {len(result.folds)}")
        print(f"  Positive OOS folds: {positive_folds}/{len(result.folds)}")
        print(f"  Median log return: {median_log_return:.3f}")
        
        # Store only serializable data
        results[hp] = {
            'median_log_return': median_log_return,
            'positive_folds': positive_folds,
            'fold_returns': fold_returns,
            'n_folds': len(result.folds),
        }
    
    # Analyze holding period sensitivity
    print("\n=== Holding Period Sensitivity Analysis ===")
    hp_values = []
    median_log_returns = []
    
    for hp, data in results.items():
        hp_values.append(hp)
        median_log_returns.append(data['median_log_return'])
        print(f"  HP={hp:2d}: median log return = {data['median_log_return']:+.3f} "
              f"(positive folds: {data['positive_folds']})")
    
    # Check if holding period variation survives
    baseline_hp = 1
    baseline_result = results[baseline_hp]['median_log_return']
    
    print(f"\n=== Holding Period Robustness ===")
    print(f"Baseline (HP={baseline_hp}): median log return = {baseline_result:+.3f}")
    
    # Check if other holding periods show similar or better results
    robust_holding_periods = []
    for hp in holding_periods:
        if hp == baseline_hp:
            continue
        result = results[hp]['median_log_return']
        # Consider robust if within 20% of baseline and positive
        tolerance = abs(baseline_result) * 0.2
        if result >= 0 and abs(result - baseline_result) <= tolerance:
            robust_holding_periods.append(hp)
            print(f"  HP={hp:2d}: median log return = {result:+.3f} - WITHIN TOLERANCE")
        else:
            print(f"  HP={hp:2d}: median log return = {result:+.3f} - OUTSIDE TOLERANCE")
    
    # Determine final verdict
    if len(robust_holding_periods) >= len(holding_periods) * 0.5:  # At least half are robust
        verdict = "ROBUST"
        print(f"\nVerdict: HOLDING PERIOD SENSITIVITY IS ROBUST - {len(robust_holding_periods)}/{len(holding_periods)-1} periods maintained edge")
    else:
        verdict = "SENSITIVE"
        print(f"\nVerdict: HOLDING PERIOD SENSITIVITY IS NOT ROBUST - only {len(robust_holding_periods)}/{len(holding_periods)-1} periods maintained edge")
    
    # Save results
    save_results = {
        'dataset_id': 'synthetic-momentum-holding-period-test',
        'lookback': 5,
        'holding_periods_tested': holding_periods,
        'baseline_holding_period': baseline_hp,
        'results': results,
        'robust_holding_periods': robust_holding_periods,
        'verdict': verdict,
        'test_parameters': {
            'train_window': 252,
            'test_window': 84,
            'warmup': 60,
            'overlap': 60,
        }
    }
    
    return save_results


if __name__ == "__main__":
    print("Testing lookback-5 momentum signal with holding period grid...")
    results = test_holding_period_grid()
    
    # Save to artifact
    output_path = Path(__file__).parent.parent / "state" / "check_artifacts" / "momentum_holding_period_results.json"
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