"""Volatility-gated momentum test (lookback-5 momentum with past-only volatility gate).

This test implements the primary learning question: "Does a single past-only
volatility gate create a predeclared robustness improvement for the qualified
momentum effect?"

Implementation:
- Use the existing momentum_signals with lookback=5
- Apply a volatility gate using conditional_signal wrapper
- Volatility gate: allow trading only when recent realized volatility < 0.20
- Run regime_stability stress testing through the existing framework
- Compare against baseline momentum (un-gated) and null benchmarks
- Use predeclared threshold without search/optimization
"""
from __future__ import annotations

import sys
from pathlib import Path
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def volatility_gate_condition(early_vol: float) -> bool:
    """Trade only when recent volatility is below the predeclared threshold.
    
    This is the predeclared past-only volatility gate: realized vol < 20% annualized.
    The threshold 0.20 is declared before execution (out_of_scope constraint).
    """
    return early_vol < 0.20


def volatility_gated_momentum_signals(closes: np.ndarray, lookback: int = 5):
    """Momentum signals gated by past-only volatility condition.
    
    This wraps the baseline momentum_signals with a volatility gate,
    so trading only occurs when recent realized volatility is low.
    """
    gated_fn = bt.conditional_signal(
        bt.momentum_signals,
        volatility_gate_condition
    )
    return gated_fn(closes, lookback=lookback)


def main():
    np.random.seed(42)
    print("=== Volatility-gated momentum test (lookback-5) ===")
    print()
    
    # Generate synthetic data with regime-switching
    bars = bt.generate_bars(
        2500,
        regimes=[
            bt.Regime(drift_annual=0.03, vol_annual=0.18, intraday_range_scale=0.015),
            bt.Regime(drift_annual=-0.01, vol_annual=0.35, intraday_range_scale=0.030),
        ],
        p_transition=0.008,
        start_price=100.0,
        seed=42,
    )
    
    # Baseline momentum (ungated) for comparison
    print("--- Baseline momentum (ungated) ---")
    baseline_signals = bt.momentum_signals(bars.closes_array(), lookback=5)
    
    baseline_grid = bt.parameter_grid_around((("lookback", 5.0),))
    baseline_result = bt.regime_stress(
        signals_fn=lambda c, **p: bt.momentum_signals(c, **p),
        bars=list(bars),
        param_grid=baseline_grid,
        baseline=("lookback", 5.0),
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
    )
    print(baseline_result.inspect())
    print()
    
    # Volatility-gated momentum
    print("--- Volatility-gated momentum (vol < 20% annualized) ---")
    gated_signals = volatility_gated_momentum_signals(bars.closes_array(), lookback=5)
    
    # Check signal integrity (no look-ahead)
    bt.check_signal_integrity(gated_signals, [b.date for b in bars], warmup=0)
    
    gated_grid = bt.parameter_grid_around((("lookback", 5.0),))
    gated_result = bt.regime_stress(
        signals_fn=lambda c, **p: volatility_gated_momentum_signals(c, **p),
        bars=list(bars),
        param_grid=gated_grid,
        baseline=("lookback", 5.0),
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
    )
    print(gated_result.inspect())
    print()
    
    # Independent walk-forward validation for the gated strategy
    print("--- Independent walk-forward validation (OOS only) ---")
    cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=60)
    wf_result = bt.walk_forward(
        list(bars),
        gated_signals,
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
        cfg=cfg,
    )
    
    print(f"Walk-forward results: {wf_result.aggregate_metrics['n_folds']} folds,")
    print(f"  mean log return: {wf_result.aggregate_metrics['mean_log_total_return']:.3f}")
    print(f"  median log return: {wf_result.aggregate_metrics['median_log_total_return']:.3f}")
    print(f"  positive OOS folds: {sum(1 for f in wf_result.folds if f.metrics['total_return'] > 0)}/{len(wf_result.folds)}")
    print()
    
    # Compare gated vs ungated performance
    print("--- Comparison: Gated vs Ungated ---")
    print(f"Baseline momentum verdict: {baseline_result.overall_verdict}")
    print(f"Gated momentum verdict:    {gated_result.overall_verdict}")
    
    baseline_medians = [s.baseline_median_log_return for s in baseline_result.scenarios]
    gated_medians = [s.baseline_median_log_return for s in gated_result.scenarios]
    
    print(f"Baseline medians across regimes: {[f'{m:.3f}' for m in baseline_medians]}")
    print(f"Gated medians across regimes:    {[f'{m:.3f}' for m in gated_medians]}")
    
    # Calculate improvement
    baseline_disp = baseline_result.candidate_dispersion
    gated_disp = gated_result.candidate_dispersion
    
    print(f"Baseline candidate dispersion: {baseline_disp:.3f}")
    print(f"Gated candidate dispersion:    {gated_disp:.3f}")
    
    if gated_disp < baseline_disp:
        improvement = (baseline_disp - gated_disp) / baseline_disp
        print(f"Dispersion improvement: {improvement:.1%}")
    else:
        print(f"Dispersion change: {gated_disp - baseline_disp:+.3f}")
    
    print()
    print("NOTE: data is synthetic and intended only for tooling validation.")
    print("This test demonstrates the predeclared volatility gate on lookback-5 momentum.")


if __name__ == "__main__":
    main()
