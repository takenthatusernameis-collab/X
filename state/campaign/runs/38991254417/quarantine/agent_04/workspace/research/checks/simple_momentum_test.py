#!/usr/bin/env python3
"""
Simple test for momentum signals with holding periods.

This creates a clear demonstration of how holding periods affect momentum signal behavior.
"""

import sys
from pathlib import Path
import numpy as np

# Add the repository root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

# Import the momentum_signals function
from research.backtest.regime_stability import momentum_signals

# Create a simple price series that will show different momentum signals
np.random.seed(42)
n = 50
closes = np.array([
    100.0, 101.0, 102.0, 103.0, 104.0,  # Up trend
    103.0, 102.0, 101.0, 100.0, 99.0,  # Down trend
    98.0, 97.0, 96.0, 95.0, 94.0,  # Down trend
    95.0, 96.0, 97.0, 98.0, 99.0,  # Up trend
    100.0, 101.0, 102.0, 103.0, 104.0,  # Up trend
    103.0, 102.0, 101.0, 100.0, 99.0,  # Down trend
    98.0, 97.0, 96.0, 95.0, 94.0,  # Down trend
    95.0, 96.0, 97.0, 98.0, 99.0,  # Up trend
    100.0, 101.0, 102.0, 103.0, 104.0,  # Up trend
    103.0, 102.0, 101.0, 100.0, 99.0,  # Down trend
    98.0, 97.0, 96.0, 95.0, 94.0,  # Down trend
    95.0, 96.0, 97.0, 98.0, 99.0,  # Up trend
    100.0, 101.0, 102.0, 103.0, 104.0,  # Up trend
    103.0, 102.0, 101.0, 100.0, 99.0,  # Down trend
    98.0, 97.0, 96.0, 95.0, 94.0,  # Down trend
    95.0, 96.0, 97.0, 98.0, 99.0,  # Up trend
    100.0, 101.0, 102.0, 103.0, 104.0,  # Up trend
    103.0, 102.0, 101.0, 100.0, 99.0,  # Down trend
    98.0, 97.0, 96.0, 95.0, 94.0,  # Down trend
    95.0, 96.0, 97.0, 98.0, 99.0,  # Up trend
    100.0, 101.0, 102.0, 103.0, 104.0,  # Up trend
    103.0, 102.0, 101.0, 100.0, 99.0,  # Down trend
    98.0, 97.0, 96.0, 95.0, 94.0,  # Down trend
    95.0, 96.0, 97.0, 98.0, 99.0,  # Up trend
    100.0, 101.0, 102.0, 103.0, 104.0,  # Up trend
    103.0, 102.0, 101.0, 100.0, 99.0,  # Down trend
    98.0, 97.0, 96.0, 95.0, 94.0,  # Down trend
    95.0, 96.0, 97.0, 98.0, 99.0,  # Up trend
    100.0, 101.0, 102.0, 103.0, 104.0,  # Up trend
    103.0, 102.0, 101.0, 100.0, 99.0,  # Down trend
    98.0, 97.0, 96.0, 95.0, 94.0,  # Down trend
    95.0, 96.0, 97.0, 98.0, 99.0,  # Up trend
    100.0, 101.0, 102.0, 103.0, 104.0,
], dtype=np.float64)

print("Price series created")

# Test 1: Simple momentum (1-day hold)
print("\n=== Testing momentum with holding periods ===")

# Test with our data
print("\nMomentum signals (lookback=5):")
for i, sig in enumerate(momentum_signals(closes, 5)):
    if i >= 5:  # After lookback period
        weight_str = f"{sig.weight:+.1f}" if sig.weight > 0 else "-.1f"
        print(f"  Bar {i+1}: weight={weight_str}")

# Now test holding periods
print("\n=== Testing holding period implementation ===")

def momentum_with_holding_period_simple(closes, lookback=5, holding_period=3):
    """Momentum signals with holding period - signals stay active for specified days."""
    n = len(closes)
    signals = [bt.Signal(i + 1, 0.0) for i in range(n)]
    
    if holding_period <= 0:
        return signals
    
    # Generate base momentum signals
    base_momentum = momentum_signals(closes, lookback)
    
    # Apply holding period logic
    position_active = None
    position_weight = 0.0
    
    for i in range(n):
        if i >= lookback:
            base_weight = base_momentum[i].weight
            
            if position_active is None:
                # Start new position
                position_active = i
                position_weight = base_weight
                signals[i] = bt.Signal(i + 1, position_weight)
            
            elif position_active is not None:
                # Continuing active position (within holding period)
                if i - position_active < holding_period:
                    # Still holding
                    signals[i] = bt.Signal(i + 1, position_weight)
                else:
                    # Holding period expired, start new position
                    position_active = i
                    position_weight = base_weight
                    signals[i] = bt.Signal(i + 1, position_weight)
            else:
                # No active position
                signals[i] = bt.Signal(i + 1, 0.0)
    
    return signals

# Test the holding period implementation
print("\nMomentum with holding period 3:")
closes_test = closes.copy()
signals_hold = momentum_with_holding_period_simple(closes_test, lookback=5, holding_period=3)

print("Key observation: Signals should stay active for 3 bars after first appearance")
for i in range(10, 30):  # Focus on bars after lookback period
    if signals_hold[i].weight != 0.0:
        weight_str = f"{signals_hold[i].weight:+.1f}" if signals_hold[i].weight > 0 else "-.1f"
        print(f"  Bar {i+1}: weight={weight_str} (position started at {i-2})")

print("\n=== Analysis ===")
print("This demonstrates how holding periods work:")
print("1. Generate momentum signal based on lookback period")
print("2. Hold that position for the specified holding period")
print("3. After holding period expires, generate new momentum signal")
print("4. Repeat the process")

print("\n=== Test completed ===")