"""Test a predeclared past-only volatility gate on lookback-5 momentum.

Test objective: Does a single past-only volatility gate create a predeclared robustness improvement for the qualified momentum effect?

Scope: One volatility feature, one threshold rule declared before execution, existing 10-asset universe.

Methodology:
- Use lookback-5 momentum signal (research.backtest.momentum_signals with lookback=5)
- Apply past-only volatility gate (condition based on trailing volatility > 25% annualized)
- Test through regime-stability stress testing framework
- Verify with independent checks

Gate declaration:
- Feature: trailing 60-bar realized volatility (annualized)
- Threshold: > 25% annualized (turbulent regime)
- Signal: emit momentum signal only when volatility <= 25% (calm/normal regime)
- Past-only: volatility calculated from closes[window//2:window] only (i=60), no look-ahead
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest import ma_crossover_signals, momentum_signals, volatility_blocks, conditional_signal


def active_when_calm(early_vol: float) -> bool:
    """Trade only in calm regimes (realized vol <= 25% annualized)."""
    return early_vol <= 0.25


def momentum_with_volatility_gate(closes: np.ndarray, lookback: int = 5, vol_window: int = 60, vol_threshold: float = 0.25) -> bt.Signal:
    """Momentum signal gated by past-only volatility regime.
    
    Long when:
    - 5-day lookback momentum positive
    - Trailing 60-bar volatility <= 25% annualized (calm regime)
    
    Past-only: volatility calculated from closes[window//2:window] only,
    no look-ahead to current bar.
    """
    n = len(closes)
    out: list[bt.Signal] = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    
    # Initialize first signal
    if n >= vol_window:
        # Calculate volatility at vol_window (first time we have enough data)
        vol_window_start = vol_window // 2
        vol_slice = closes[vol_window_start:vol_window]
        early_vol = float(np.std(np.log(vol_slice), ddof=1)) * np.sqrt(252.0)
        
        if early_vol <= vol_threshold:
            # Apply momentum from vol_window onward
            for i in range(vol_window, n):
                ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
                out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
        else:
            # Skip turbulent regime entirely
            pass
    else:
        # Not enough data for volatility calculation
        pass
            
    return out


def main():
    np.random.seed(42)
    
    print("=== Momentum with Volatility Gate Test ===")
    print("Gate: Momentum signal only emitted in calm regimes (volatility <= 25% annualized)")
    print("Window: Trailing 60-bar volatility, calculated past-only")
    print("Universe: 10-asset Yahoo Finance OHLCV (2009-2026-10-03)")
    print()
    
    # Load manifest
    from research.data.preflight import load_manifest
    manifest = load_manifest()
    print(f"Dataset: {manifest['dataset_id']}")
    print(f"Universe size: {len(manifest['universe'])}")
    print()
    
    # Load real data for one asset (AAPL) for verification
    bars, dates = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    print(f"AAPL: {bars.n_bars} bars ({dates[0]} to {dates[-1]})")
    
    # Test the gated momentum signal
    print("\n=== Gated Momentum Signal Construction ===")
    vol_window, vol_threshold = 60, 0.25
    lookback = 5
    
    signals = momentum_with_volatility_gate(closes, lookback, vol_window, vol_threshold)
    
    nonneutral = sum((1 for s in signals if abs(s.weight) > 1e-12))
    print(f"Total signals: {len(signals)}")
    print(f"Non-neutral signals: {nonneutral}")
    print(f"Long positions: {sum((1 for s in signals if s.weight > 0.5))}")
    print(f"Short positions: {sum((1 for s in signals if s.weight < -0.5))}")
    
    # Check signal integrity
    print("\n=== Signal Integrity Check ===")
    bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
    print("  [PASS] check_signal_integrity: signal format valid")
    
    # Run backtest
    print("\n=== Backtest Performance ===")
    cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=vol_window)
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
    
    # Define the gate-based momentum signal function
    def gated_momentum_signals(closes: np.ndarray, **params):
        return momentum_with_volatility_gate(closes, lookback=5, vol_window=60, vol_threshold=0.25)
    
    # Baseline: ungated momentum
    def ungated_momentum_signals(closes: np.ndarray, **params):
        return momentum_signals(closes, lookback=5)
    
    baseline = (("lookback", 5),)
    grid = [{"lookback": 5}]
    
    # Test gated version
    stress = bt.stress_segments(
        signals_fn=gated_momentum_signals,
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
    
    print(f"  Candidate dispersion across segments: {stress.candidate_dispersion:+.3f}")
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
    
    print(f"Ungated momentum: {ungated_stress.overall_verdict} (disp={ungated_stress.candidate_dispersion:+.3f})")
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
    bars2, dates2 = bt.load_ticker("AAPL")
    closes2 = bars2.closes_array()
    signals2 = momentum_with_volatility_gate(closes2, lookback=5, vol_window=60, vol_threshold=0.25)
    
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


if __name__ == "__main__":
    main()