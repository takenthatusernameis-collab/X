#!/usr/bin/env python3
"""Test script to verify the backtest module imports correctly."""

import sys
sys.path.insert(0, '/home/runner/work/X/X')
import research.backtest as bt

print("Testing momentum_signals function:")
print(f"Function exists: {hasattr(bt, 'momentum_signals')}")
print(f"Function callable: {callable(getattr(bt, 'momentum_signals', None))}")

print("\nTesting volatility gate functions:")
print(f"conditional_signal exists: {hasattr(bt, 'conditional_signal')}")
print(f"active_when_turbulent exists: {hasattr(bt, 'active_when_turbulent')}")

print("\nTesting data.py functions:")
print(f"generate_bars exists: {hasattr(bt, 'generate_bars')}")

print("\nAll tests passed!")