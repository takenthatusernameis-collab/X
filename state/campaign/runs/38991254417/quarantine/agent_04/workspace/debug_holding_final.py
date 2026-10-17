#!/usr/bin/env python3
"""
Debug momentum signal with holding period implementation.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

# Generate data that shows different momentum signals
# Use a sequence that will produce alternating LONG/SHORT signals
np.random.seed(123)
# Create a sequence with clear turning points
closes = np.array([
    100.0, 101.0, 102.0, 103.0, 104.0,  # Upward trend
    103.0, 102.0, 101.0, 100.0, 99.0,    # Downward trend  
    98.0, 97.0, 96.0, 95.0, 94.0,       # Downward trend
    95.0, 96.0, 97.0, 98.0, 99.0,       # Upward trend
] * 5 + [  # Repeat to get enough bars
    100.0, 101.0, 100.0, 101.0, 100.0,  # Oscillating
] * 5)

print("First 30 closes:", closes[:30])
print("\nPrice changes (i - i-1):", closes[1:31] - closes[:30])

# Generate the signals
base_signals = bt.momentum_signals(closes, lookback=5)
print("\n=== Base momentum signals (first 30 bars) ===")
for i in range(30):
    if i >= 5:  # After lookback period
        print(f"Bar {i+1}: weight={base_signals[i].weight} (lookback return: {'UP' if base_signals[i].weight > 0 else 'DOWN'})")

# Test with holding period implementation
def momentum_with_holding_period_simple(closes, lookback, holding_period):
    """Simple implementation that creates positions held for specified period."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    
    if holding_period <= 0:
        return out
    
    # Generate base momentum signals
    base_signals = bt.momentum_signals(closes, lookback=lookback)
    
    # Create active position tracker
    position_start_date = None
    position_weight = 0.0
    
    for i in range(n):
        # Check if we need to start a new position
        # Start if we have a base signal and either:
        # 1. No active position, or
        # 2. Current holding period has expired
        if base_signals[i].weight != 0 and (position_start_date is None or (i - position_start_date) >= holding_period):
            # Start new position
            position_start_date = i
            position_weight = base_signals[i].weight
            out[i] = bt.Signal(date=i + 1, weight=position_weight)
        
        # If we have an active position and holding period hasn't expired, maintain it
        elif position_start_date is not None and (i - position_start_date) < holding_period:
            # Continue holding
            out[i] = bt.Signal(date=i + 1, weight=position_weight)
        
        # Otherwise stay neutral (either no position or position expired)
        else:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
    
    return out

print("\n=== Testing holding period 1 ===")
signals_1 = momentum_with_holding_period_simple(closes, lookback=5, holding_period=1)
for i in range(30):
    print(f"Bar {i+1}: weight={signals_1[i].weight}")

print("\n=== Testing holding period 3 ===")
signals_3 = momentum_with_holding_period_simple(closes, lookback=5, holding_period=3)
for i in range(30):
    print(f"Bar {i+1}: weight={signals_3[i].weight}")

print("\n=== Testing holding period 5 ===")
signals_5 = momentum_with_holding_period_simple(closes, lookback=5, holding_period=5)
for i in range(30):
    print(f"Bar {i+1}: weight={signals_5[i].weight}")