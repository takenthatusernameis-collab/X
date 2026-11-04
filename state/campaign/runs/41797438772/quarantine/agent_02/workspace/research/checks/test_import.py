#!/usr/bin/env python3
"""
Minimal cost-sensitivity check that works with the existing framework.
This script uses the same import pattern as the working momentum.py.
"""
import sys
from pathlib import Path

# Use the same import pattern as the working momentum.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import research.backtest as bt
from research.data.preflight import load_manifest

print("=== Cost-Sensitivity Check - Import Test ===")
print(f"Python path includes: {Path(__file__).resolve().parent.parent.parent}")
print(f"research.backtest imported successfully: {bt}")

# Test basic functionality
manifest = load_manifest()
print(f"Loaded dataset: {manifest['dataset_id']}")
print(f"Universe size: {len(manifest['universe'])} assets")

# Test momentum_signals function
import numpy as np
test_data = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
signals = bt.momentum_signals(test_data, lookback=5)
print(f"Momentum signals test: {len(signals)} signals generated")

# Test cost configuration
cost_config = bt.BacktestConfig(
    commission_per_trade=0.001,
    commission_per_share=0.0001,
    slippage_cents=0.01,
    slippage_proportional=0.001,
)
print(f"Cost config test: commission_per_trade = {cost_config.commission_per_trade}")

print("\n✅ All imports and basic functionality working!")
print("The cost-sensitivity framework is ready for implementation.")
