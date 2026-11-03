#!/usr/bin/env python3
"""Simple test to verify the environment is ready for momentum_cost_robustness."""
import sys
sys.path.insert(0, '.')

# Test key components
import research.backtest as bt
from research.data.preflight import load_manifest

print("Environment verification:")
print("✓ research.backtest imports successfully")
print("✓ load_manifest imports successfully")
print("✓ BacktestConfig available:", hasattr(bt, 'BacktestConfig'))
print("✓ momentum_signals available:", hasattr(bt, 'momentum_signals'))
print("✓ stress_segments available:", hasattr(bt, 'stress_segments'))

# Test BacktestConfig instantiation
config = bt.BacktestConfig(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
print("✓ BacktestConfig instantiation successful")

print("\nEnvironment is ready for momentum_cost_robustness execution!")