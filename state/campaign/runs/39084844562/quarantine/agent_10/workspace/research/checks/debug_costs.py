"""Debug script to understand how costs are being applied."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

# Create configs with different costs
no_cost_config = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=60,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)

real_cost_config = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.003,
    commission_per_share=0.000001,
    slippage_cents=0.05,
    slippage_proportional=0.0005,
    warmup_periods=60,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)

print("=== Config comparison ===")
print(f"No cost config:")
print(f"  commission_per_trade: {no_cost_config.commission_per_trade}")
print(f"  commission_per_share: {no_cost_config.commission_per_share}")
print(f"  slippage_cents: {no_cost_config.slippage_cents}")
print(f"  slippage_proportional: {no_cost_config.slippage_proportional}")
print(f"\nReal cost config:")
print(f"  commission_per_trade: {real_cost_config.commission_per_trade}")
print(f"  commission_per_share: {real_cost_config.commission_per_share}")
print(f"  slippage_cents: {real_cost_config.slippage_cents}")
print(f"  slippage_proportional: {real_cost_config.slippage_proportional}")

# Test with a simple single asset to see if costs are applied
manifest = load_manifest()
ticker = "AAPL"

# Load the ticker
bars, dates = bt.load_ticker(ticker)

# Generate a simple signal (momentum with lookback 5)
closes = bars.closes_array()
signals = bt.momentum_signals(closes, lookback=5)

print(f"\n=== Testing with {ticker} ===")
print(f"Number of bars: {len(bars)}")
print(f"Number of signals: {len(signals)}")

# Test no cost config
print(f"\n=== No cost config ===")
print(f"Config commission_per_trade: {no_cost_config.commission_per_trade}")
print(f"Config commission_per_share: {no_cost_config.commission_per_share}")

# Test real cost config
print(f"\n=== Real cost config ===")
print(f"Config commission_per_trade: {real_cost_config.commission_per_trade}")
print(f"Config commission_per_share: {real_cost_config.commission_per_share}")
