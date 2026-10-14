"""Simple test to understand the baseline format issue."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def test_baseline_format():
    print("Testing baseline format...")
    
    # Generate synthetic data
    bars = bt.generate_bars(100, seed=42)
    
    # Test with single parameter
    print("\n1. Testing with single parameter:")
    grid = bt.parameter_grid_around((("lookback", 5.0),))
    print(f"   Grid: {grid}")
    print(f"   Grid type: {type(grid)}")
    
    # Test with baseline as tuple of tuples
    baseline1 = (("lookback", 5.0),)
    print(f"   Baseline1: {baseline1}")
    print(f"   Baseline1 type: {type(baseline1)}")
    print(f"   dict(baseline1): {dict(baseline1)}")
    
    # Test with baseline as tuple of 2 elements
    baseline2 = ("lookback", 5.0)
    print(f"   Baseline2: {baseline2}")
    print(f"   Baseline2 type: {type(baseline2)}")
    try:
        print(f"   dict(baseline2): {dict(baseline2)}")
    except Exception as e:
        print(f"   dict(baseline2) ERROR: {e}")
    
    # Test the exact function call that's failing
    print("\n2. Testing regime_stress with baseline1...")
    try:
        result = bt.regime_stress(
            signals_fn=lambda c, **p: bt.momentum_signals(c, **p),
            bars=list(bars),
            param_grid=grid,
            baseline=baseline1,
            train_window=20,
            test_window=10,
            warmup=20,
            overlap_window=10,
        )
        print(f"   SUCCESS! Result: {result.overall_verdict}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Testing regime_stress with baseline2...")
    try:
        result = bt.regime_stress(
            signals_fn=lambda c, **p: bt.momentum_signals(c, **p),
            bars=list(bars),
            param_grid=grid,
            baseline=baseline2,
            train_window=20,
            test_window=10,
            warmup=20,
            overlap_window=10,
        )
        print(f"   SUCCESS! Result: {result.overall_verdict}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_baseline_format()
