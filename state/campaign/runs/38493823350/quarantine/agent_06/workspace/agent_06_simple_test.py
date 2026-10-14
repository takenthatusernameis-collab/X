"""Simplified volatility-gated momentum test."""
from __future__ import annotations

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def volatility_gate_condition(early_vol: float) -> bool:
    """Trade only when recent volatility is below the predeclared threshold.
    
    This is the predeclared past-only volatility gate: realized vol < 20% annualized.
    The threshold 0.20 is declared before execution (out_of_scope constraint).
    """
    return early_vol < 0.20


def volatility_gated_momentum_signals(closes, lookback: int = 5):
    """Momentum signals gated by past-only volatility condition."""
    gated_fn = bt.conditional_signal(
        bt.momentum_signals,
        volatility_gate_condition
    )
    return gated_fn(closes, lookback=lookback)


def main():
    print("=== Simplified volatility-gated momentum test ===")
    print()
    
    # Generate synthetic data
    bars = bt.generate_bars(1000, seed=42)
    
    # Test baseline momentum (ungated)
    print("--- Baseline momentum (ungated) ---")
    baseline_signals = bt.momentum_signals(bars.closes_array(), lookback=5)
    
    # Correct baseline format: tuple of tuples
    baseline = (("lookback", 5.0),)
    baseline_grid = bt.parameter_grid_around(baseline)
    
    baseline_result = bt.regime_stress(
        signals_fn=lambda c, **p: bt.momentum_signals(c, **p),
        bars=list(bars),
        param_grid=baseline_grid,
        baseline=baseline,
        train_window=84,
        test_window=28,
        warmup=20,
        overlap_window=10,
    )
    print(f"Baseline verdict: {baseline_result.overall_verdict}")
    print()
    
    # Test volatility-gated momentum
    print("--- Volatility-gated momentum (vol < 20% annualized) ---")
    gated_signals = volatility_gated_momentum_signals(bars.closes_array(), lookback=5)
    
    # Check signal integrity
    bt.check_signal_integrity(gated_signals, [b.date for b in bars], warmup=0)
    
    gated_result = bt.regime_stress(
        signals_fn=lambda c, **p: volatility_gated_momentum_signals(c, **p),
        bars=list(bars),
        param_grid=baseline_grid,
        baseline=baseline,
        train_window=84,
        test_window=28,
        warmup=20,
        overlap_window=10,
    )
    print(f"Gated verdict: {gated_result.overall_verdict}")
    print()
    
    # Compare results
    print("--- Comparison ---")
    print(f"Baseline medians: {[f'{m:.3f}' for m in [s.baseline_median_log_return for s in baseline_result.scenarios]]}")
    print(f"Gated medians:    {[f'{m:.3f}' for m in [s.baseline_median_log_return for s in gated_result.scenarios]]}")
    
    print(f"\nBaseline dispersion: {baseline_result.candidate_dispersion:.3f}")
    print(f"Gated dispersion:    {gated_result.candidate_dispersion:.3f}")
    
    # Calculate improvement
    if gated_result.candidate_dispersion < baseline_result.candidate_dispersion:
        improvement = (baseline_result.candidate_dispersion - gated_result.candidate_dispersion) / baseline_result.candidate_dispersion
        print(f"Dispersion improvement: {improvement:.1%}")
    else:
        change = gated_result.candidate_dispersion - baseline_result.candidate_dispersion
        print(f"Dispersion change: {change:+.3f}")
    
    print("\n=== Test completed successfully ===")


if __name__ == "__main__":
    main()
