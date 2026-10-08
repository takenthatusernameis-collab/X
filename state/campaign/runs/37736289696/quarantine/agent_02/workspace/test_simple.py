"""Simple momentum cost robustness test for debugging."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

# Realistic cost model from examples/ma_crossover.py
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

def momentum_signals_for_lookback(closes: list[float], lookback: int) -> list[bt.Signal]:
    """Generate momentum signals for a given lookback period."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback - 1, n):
        prev_return = (closes[i] - closes[i - lookback]) / closes[i - lookback]
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if prev_return > 0 else -1.0)
    return signals

def main():
    print("=== Simple Momentum Cost Robustness Test ===")
    
    # Test with synthetic data
    bars = bt.generate_bars(300, seed=42)
    closes = bars.closes_array()
    
    # Test lookback 5
    lookback = 5
    signals = momentum_signals_for_lookback(closes, lookback)
    
    # Verify signal integrity
    bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
    
    # Create labels for segment_fn
    labels = ["turbulent" if i % 2 == 0 else "calm" for i in range(len(closes))]
    segment_fn = bt.segment_fn_from_labels(labels)
    
    # Run with realistic costs
    res = bt.stress_segments(
        signals_fn=lambda c, **params: momentum_signals_for_lookback(c, params.get("lookback", lookback)),
        bars=list(bars),
        signals=signals,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=segment_fn,
        train_window=60,
        test_window=60,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        min_segment_bars=1,
        cfg=REALISTIC_COST_CONFIG,
    )
    
    print(f"Test completed successfully!")
    print(f"Overall verdict: {res.overall_verdict}")
    print(f"Number of folds: {res.n_folds}")
    print(f"Candidate dispersion: {res.candidate_dispersion}")
    print(f"Null dispersion: {res.null_dispersion}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
