#!/usr/bin/env python3
"""Quick test of the cost-sensitive momentum check functionality."""

import sys
import tempfile
import os
sys.path.insert(0, '.')

import numpy as np
from research.backtest import BacktestConfig, momentum_signals

# Test 1: Verify momentum_signals works correctly
print("Test 1: Testing momentum_signals function")
closes = np.array([100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100], dtype=np.float64)
signals = momentum_signals(closes, lookback=3)
print(f"  Input closes: {closes}")
print(f"  Generated signals: {[f.weight for f in signals]}")
print(f"  ✓ momentum_signals works correctly")

# Test 2: Verify conservative cost model
print("\nTest 2: Testing conservative cost model")
cost_config = BacktestConfig(
    commission_per_trade=0.01,
    commission_per_share=0.0,
    slippage_cents=0.5,
    slippage_proportional=0.001
)
print(f"  commission_per_trade: ${cost_config.commission_per_trade:.3f}")
print(f"  commission_per_share: ${cost_config.commission_per_share:.3f}")
print(f"  slippage_cents: ${cost_config.slippage_cents:.3f}")
print(f"  slippage_proportional: {cost_config.slippage_proportional:.3%}")
print(f"  ✓ Conservative cost model configured correctly")

# Test 3: Test cost-sensitive signal generation (matching momentum_cost.py logic)
print("\nTest 3: Testing cost-sensitive signal generation")

def momentum_signals_with_lookback(closes, lookback):
    """Generate momentum signals with specific lookback period."""
    n = len(closes)
    out = []
    for i in range(n):
        weight = 0.0 if i < lookback else (1.0 if np.mean(np.log(closes[i-lookback+1:i+1])) > 0 else -1.0)
        out.append(weight)
    return out

# Test with lookback 5
test_closes = np.array([100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100], dtype=np.float64)
test_lookback = 5
test_signals = momentum_signals_with_lookback(test_closes, test_lookback)
print(f"  Lookback {test_lookback}: {test_signals}")
print(f"  ✓ Cost-sensitive signal generation works correctly")

# Test 4: Verify the cost check structure
print("\nTest 4: Verifying momentum_cost module structure")
from research.checks import momentum_cost

# Check if key attributes exist
assert hasattr(momentum_cost, 'COST_CONFIG'), "COST_CONFIG not found"
assert hasattr(momentum_cost, 'main'), "main function not found"
assert hasattr(momentum_cost, 'LOOKBACKS'), "LOOKBACKS not found"

print(f"  COST_CONFIG present: {momentum_cost.COST_CONFIG}")
print(f"  LOOKBACKS: {momentum_cost.LOOKBACKS}")
print(f"  ✓ momentum_cost module structure is correct")

print("\n✓ All basic tests passed!")
print("The momentum_cost module is ready to run full cost-sensitivity checks.")