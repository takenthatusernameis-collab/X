#!/usr/bin/env python3
"""Test volatility-normalized sizing implementation directly."""

import sys
import numpy as np
import json
from pathlib import Path

# Add the project to the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import research.backtest as bt
from research.checks.momentum import volatility_normalizing_sizing, volatility_normalized_momentum_signals

def test_volatility_normalizing_sizing():
    """Test the volatility-normalized sizing function."""
    print("Testing volatility-normalized sizing function...")
    
    # Generate some synthetic price data
    n_bars = 252
    prices = np.linspace(100, 150, n_bars)
    np.random.seed(42)
    for i in range(1, n_bars):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.001, 0.02))
    
    # Test the sizing function
    sizing = volatility_normalizing_sizing(prices, window=60, target_vol=0.20, max_scaling=2.0, min_scaling=0.1)
    
    print(f"Generated sizing array of length {len(sizing)}")
    print(f"Min sizing: {np.min(sizing):.4f}, Max sizing: {np.max(sizing):.4f}")
    print(f"Mean sizing: {np.mean(sizing):.4f}")
    
    # Test signal integration
    signals = volatility_normalized_momentum_signals(prices, lookback=5, vol_window=60, vol_target=0.20)
    print(f"Generated momentum signals with volatility-normalized sizing")
    print(f"Number of signals: {len(signals)}")
    
    # Check that signals have non-zero weights
    non_zero_signals = sum(1 for s in signals if abs(s.weight) > 0)
    print(f"Signals with non-zero weights: {non_zero_signals}")
    
    if non_zero_signals == 0:
        print("ERROR: All signals have zero weight!")
        return False
    
    print("SUCCESS: Volatility-normalized sizing test passed!")
    return True

def main():
    """Main test function."""
    print("=== Testing Volatility-Normalized Sizing Implementation ===\n")
    
    # Test the sizing function
    if not test_volatility_normalizing_sizing():
        return 1
    
    print("\n=== All tests passed! ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())