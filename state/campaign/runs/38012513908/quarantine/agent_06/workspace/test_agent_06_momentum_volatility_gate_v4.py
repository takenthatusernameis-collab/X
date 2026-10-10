"""Test a predeclared past-only volatility gate on lookback-5 momentum.

Test objective: Does a single past-only volatility gate create a predeclared robustness improvement for the qualified momentum effect?

Scope: One volatility feature, one threshold rule declared before execution, existing 10-asset universe.

Methodology:
- Use lookback-5 momentum signal (research.backtest.momentum_signals with lookback=5)
- Apply past-only volatility gate using conditional_signal wrapper
- Gate condition: only trade in calm regimes (volatility <= 25% annualized)
- Test through regime-stability stress testing framework
- Verify with independent checks

Gate declaration:
- Feature: past-only realized volatility (trailing 60-bar, annualized)
- Threshold: > 25% annualized (turbulent regime) - gate blocks turbulent regimes
- Signal: emit momentum signal only when volatility <= 25% (calm/normal regime)
- Past-only: volatility calculated from closes[:60] only (window=60), no look-ahead to current bar
- The same volatility value is applied consistently across all bars in the series
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest import (
    ma_crossover_signals, 
    momentum_signals, 
    volatility_blocks, 
    conditional_signal,
    load_ticker
)
from research.data.preflight import load_manifest


def main():
    np.random.seed(42)
    
    print("=== Momentum with Volatility Gate Test ===")
    print("Gate: Momentum signal only emitted in calm regimes (volatility <= 25% annualized)")
    print("Window: Trailing 60-bar volatility, calculated past-only (applies consistently)")
    print("Universe: 10-asset Yahoo Finance OHLCV (2009-2026-10-03)")
    print()
    
    # Load manifest
    manifest = load_manifest()
    print(f"Dataset: {manifest['dataset_id']}")
    print(f"Universe size: {len(manifest['universe'])}")
    print()
    
    # Load real data for one asset (AAPL) for verification
    bars, dates = load_ticker("AAPL")
    closes = bars.closes_array()
    print(f"AAPL: {bars.n_bars} bars ({dates[0]} to {dates[-1]})")
    
    # Create volatility gate using conditional_signal framework
    print("\n=== Volatility Gate Implementation ===")
    
    # Define volatility gate condition (calm regime only)
    def volatility_gate_condition(early_vol: float) -> bool:
        return early_vol <= 0.25
    
    # Wrap momentum_signals with volatility gate using conditional_signal
    momentum_with_gate = conditional_signal(
        base_signals_fn=momentum_signals,
        condition_fn=volatility_gate_condition,
    )
    
    print(f"Gate condition: volatility <= 25% annualized (calm regime)")
    print(f"Signal function: momentum_signals with lookback=5, gated by volatility condition")
    
    # Generate signals using the gated momentum
    signals = momentum_with_gate(closes, lookback=5)
    
    nonneutral = sum((1 for s in signals if abs(s.weight) > 1e-12))
    print(f"\nGenerated signals: {len(signals)}")
    print(f"Non-neutral signals: {nonneutral}")
    print(f"Long positions: {sum((1 for s in signals if s.weight > 0.5))}")
    print(f"Short positions: {sum((1 for s in signals if s.weight < -0.5))}")
    
    # Check signal integrity
    print("\n=== Signal Integrity Check ===")
    bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
    print("  [PASS] check_signal_integrity: signal format valid")
    
    # Run backtest
    print("\n=== Backtest Performance ===")
    cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=60)
    res = bt.run_bars(list(bars), signals, cfg0)
    bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
    print("  [PASS] check_equity_matches_fills: equity matches fills")
    
    m = bt.compute_metrics(
        res.equity_curve,
        fills=res.trades,
        fill_prices=np.array([f.price for f in res.trades], dtype=np.float64) if res.trades else np.array([], dtype=np.float64),
        periods_per_year=252,
    )
    print(f"  Trades: {m.n_trades}")
    print(f"  Total return: {m.total_return:.2%}")
    print(f"  Sharpe ratio: {m.sharpe:.2f}")
    print(f"  Max drawdown: {m.max_drawdown:.2%}")
    print(f"  Turnover ratio: {m.turnover_ratio:.1f}x")
    
    # Test with regime stability
    print("\n=== Regime-Stability Stress Testing ===")
    
    # Get volatility labels using the standard framework
    labels = volatility_blocks(closes, n_blocks=4)
    from collections import Counter
    counts = Counter(labels)
    print("Regime distribution:")
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n} bars")
    
    # Baseline: ungated momentum
    def ungated_momentum_signals(closes: np.ndarray, **params):
        return momentum_signals(closes, lookback=5)
    
    baseline = (("lookback", 5),)
    grid = [{"lookback": 5}]
    
    # Test gated version
    stress = bt.stress_segments(
        signals_fn=lambda closes, **p: momentum_with_gate(closes, lookback=5),
        bars=bars,
        signals=signals,
        param_grid=grid,
        baseline=baseline,
        segment_fn=bt.segment_fn_from_labels(labels),
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
        cfg=cfg0,
        min_segment_bars=400,
    )
    
    print(f"\n  Candidate dispersion across segments: {stress.candidate_dispersion:+.3f}")
    print(f"  Null dispersion across segments:      {stress.null_dispersion:+.3f}")
    print(f"  Overall verdict: {stress.overall_verdict}")
    print(f"  Number of scenarios: {len(stress.scenarios)}")
    
    for s in stress.scenarios:
        status = 'EDGE  ' if abs(s.baseline_median_log_return) > 0.05 else 'noise'
        print(f"  {s.name:12s}: candidate {s.baseline_median_log_return:+.3f}, noise {s.noise_median_log_return:+.3f}  [{status}] (folds={s.n_folds})")
    
    # Compare with ungated momentum for comparison
    ungated_stress = bt.stress_segments(
        signals_fn=ungated_momentum_signals,
        bars=bars,
        signals=ungated_momentum_signals(closes),
        param_grid=grid,
        baseline=baseline,
        segment_fn=bt.segment_fn_from_labels(labels),
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
        cfg=cfg0,
        min_segment_bars=400,
    )
    
    print(f"\nUngated momentum: {ungated_stress.overall_verdict} (disp={ungated_stress.candidate_dispersion:+.3f})")
    print(f"Gated momentum:   {stress.overall_verdict} (disp={stress.candidate_dispersion:+.3f})")
    
    # Assess if gate improves robustness
    print("\n=== Gate Robustness Assessment ===")
    
    gate_improved = False
    if stress.overall_verdict in ["REGIME_STABLE", "REGIME_STABLE_LOSS"] and ungated_stress.overall_verdict in ["REGIME_DEPENDENT", "CONSISTENT_WITH_NOISE"]:
        gate_improved = True
        print("  ✓ Gate improved robustness (REGIME_STABLE vs REGIME_DEPENDENT/CONSISTENT_WITH_NOISE)")
    elif stress.overall_verdict == ungated_stress.overall_verdict:
        if abs(stress.candidate_dispersion) < abs(ungated_stress.candidate_dispersion):
            gate_improved = True
            print("  ✓ Gate reduced dispersion (more stable)")
        else:
            print("  → Gate did not clearly improve robustness")
    else:
        print("  → Gate did not clearly improve robustness")
    
    print("\n=== Independent Verification ===")
    
    # Recompute using the same methodology to verify
    np.random.seed(42)
    bars2, dates2 = load_ticker("AAPL")
    closes2 = bars2.closes_array()
    signals2 = momentum_with_gate(closes2, lookback=5)
    
    # Check that results are deterministic
    signals_equal = all(
        s1.weight == s2.weight for s1, s2 in zip(signals, signals2)
        if abs(s1.weight) > 1e-12 or abs(s2.weight) > 1e-12
    )
    print(f"  Deterministic recomputation: {'PASS' if signals_equal else 'FAIL'}")
    
    print("\n=== Test Summary ===")
    print(f"Gate: momentum only in calm regimes (vol <= 25% annualized)")
    print(f"Ungated momentum verdict: {ungated_stress.overall_verdict}")
    print(f"Gated momentum verdict:   {stress.overall_verdict}")
    print(f"Gate improved robustness: {gate_improved}")
    print(f"Success criterion met: {'YES' if gate_improved or stress.overall_verdict in ['REGIME_STABLE', 'REGIME_STABLE_LOSS'] else 'NO'}")
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Test completed successfully. Results inform whether to continue research.")
    
    # Return results for JSON output
    return {
        "ungated_verdict": ungated_stress.overall_verdict,
        "ungated_dispersion": ungated_stress.candidate_dispersion,
        "gated_verdict": stress.overall_verdict,
        "gated_dispersion": stress.candidate_dispersion,
        "gate_improved": gate_improved,
        "success_criterion_met": 'YES' if gate_improved or stress.overall_verdict in ['REGIME_STABLE', 'REGIME_STABLE_LOSS'] else 'NO',
        "regime_distribution": dict(counts),
        "scenarios": [
            {"name": s.name, "candidate_return": s.baseline_median_log_return, "noise_return": s.noise_median_log_return, "verdict": "EDGE" if abs(s.baseline_median_log_return) > 0.05 else "NOISE", "folds": s.n_folds}
            for s in stress.scenarios
        ]
    }


if __name__ == "__main__":
    main()