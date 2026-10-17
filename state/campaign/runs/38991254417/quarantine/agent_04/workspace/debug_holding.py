#!/usr/bin/env python3
"""
Debug momentum signal with holding period.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

# Generate synthetic data
np.random.seed(42)
bars = bt.generate_bars(
    n_bars=100,
    regimes=[bt.Regime(drift_annual=0.03, vol_annual=0.18)],
    start_price=100.0,
    seed=42,
)

closes = bars.closes_array()
print("First 20 closes:", closes[:20])

# Test the simple momentum signal
signals_1d = bt.momentum_signals(closes, lookback=5)
print("\n=== Simple 1-day momentum signals (first 30) ===")
for i in range(30):
    print(f"Bar {i+1}: weight={signals_1d[i].weight}")

# Now test with a custom holding period implementation
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

# Test with different holding periods
print("\n=== Testing holding period implementation ===")
for hp in [1, 2, 3, 5]:
    signals = momentum_with_holding_period_simple(closes, lookback=5, holding_period=hp)
    print(f"\nHolding period {hp} days:")
    
    # Look for position changes
    in_position = False
    for i in range(30):
        if signals[i].weight != 0.0 and not in_position:
            print(f"  Started position at bar {i+1}: weight={signals[i].weight}")
            in_position = True
        elif signals[i].weight == 0.0 and in_position:
            print(f"  Ended position at bar {i}: was holding for {i - (i-5)} days")
            in_position = False
        elif signals[i].weight != 0.0 and in_position:
            pass  # continuing position
    
    # Check if positions are being held
    print(f"  Bars with non-zero weight: {sum(1 for s in signals if s.weight != 0.0)} out of {len(signals)}")