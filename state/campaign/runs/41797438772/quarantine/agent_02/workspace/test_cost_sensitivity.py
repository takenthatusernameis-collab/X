#!/usr/bin/env python3
"""
Simple cost-sensitivity check script to test the workflow.
This is a minimal implementation to verify the concept works.
"""
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

# Simple cost sensitivity check
print("=== Simple Cost-Sensitivity Check ===")

# Load manifest
manifest = load_manifest()
print(f"Loaded dataset: {manifest['dataset_id']}")
print(f"Universe size: {len(manifest['universe'])} assets")

# Test cost model
cost_config = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.001,
    commission_per_share=0.0001,
    slippage_cents=0.01,
    slippage_proportional=0.001,
)

print("Cost model parameters:")
print(f"  commission_per_trade: ${cost_config.commission_per_trade:.4f}")
print(f"  commission_per_share: ${cost_config.commission_per_share:.6f}")
print(f"  slippage_cents: ${cost_config.slippage_cents:.2f}")
print(f"  slippage_proportional: {cost_config.slippage_proportional:.4f} ({cost_config.slippage_proportional*100:.2f} bps)")

# Test momentum_signals function
import numpy as np
test_closes = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
signals = bt.momentum_signals(test_closes, lookback=5)
print(f"\nMomentum signals test:")
print(f"  Input length: {len(test_closes)} bars")
print(f"  Output signals: {len(signals)} signals")
print(f"  Signal weights: {set(s.weight for s in signals[:5])}...")

# Verify BacktestConfig includes cost parameters
print(f"\nBacktestConfig verification:")
print(f"  Has commission_per_trade: {hasattr(cost_config, 'commission_per_trade')}")
print(f"  Has commission_per_share: {hasattr(cost_config, 'commission_per_share')}")
print(f"  Has slippage_cents: {hasattr(cost_config, 'slippage_cents')}")
print(f"  Has slippage_proportional: {hasattr(cost_config, 'slippage_proportional')}")

print(f"\n commission_per_trade: ${cost_config.commission_per_trade:.4f}")
print(f" commission_per_share: ${cost_config.commission_per_share:.6f}")
print(f" slippage_cents: ${cost_config.slippage_cents:.2f}")
print(f" slippage_proportional: {cost_config.slippage_proportional:.4f}")

print("\n=== Cost-Sensitivity Check Ready ===")
print("The workflow is designed to:")
print("1. Load and align 10 assets to common bar count")
print("2. Test lookback 3, 5, 10 momentum signals with costs")
print("3. Compare against cost-aware null benchmark")
print("4. Classify edges as surviving or disappearing under costs")
print("\nFull implementation would require running the complete workflow.")
print("The cost-sensitive framework is successfully loaded and configured.")
