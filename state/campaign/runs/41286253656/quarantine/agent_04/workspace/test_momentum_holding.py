#!/usr/bin/env python3
"""Test momentum_signals_with_holding function."""
import sys
sys.path.insert(0, '.')

import research.backtest as bt
from research.backtest.data import generate_bars

# Generate test data
bars = generate_bars(600, seed=42)
closes = bars.closes_array()

# Test momentum_signals_with_holding function with 1-day holding period
print("Testing momentum_signals_with_holding function...")
print(f"Data length: {len(closes)}")
print(f"First 10 closes: {closes[:10]}")

try:
    signals = bt.momentum_signals_with_holding(closes, 5, 1)
    print(f"Signals length: {len(signals)}")
    print(f"First 10 signals: {signals[:10]}")
    print(f"Non-zero signals: {sum(1 for s in signals if abs(s.weight) > 1e-12)}")
    print("SUCCESS: Function works!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()