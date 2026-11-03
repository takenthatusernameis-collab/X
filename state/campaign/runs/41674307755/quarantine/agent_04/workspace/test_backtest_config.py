#!/usr/bin/env python3
"""Simple test of BacktestConfig and imports."""
import sys
sys.path.insert(0, '.')
import research.backtest as bt

print('Testing basic bt module import...')
print('Testing BacktestConfig...')
cfg = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
print('cfg created successfully')
print('Commission per trade:', cfg.commission_per_trade)
print('Commission per share:', cfg.commission_per_share)
print('Testing passed!')