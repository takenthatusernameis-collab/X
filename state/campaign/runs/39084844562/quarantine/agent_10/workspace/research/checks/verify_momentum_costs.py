"""Verify momentum results with explicit cost parameters.

This script runs the momentum check again but with explicit cost parameters
to verify that the costs are being applied correctly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Test with different cost configurations
NO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=WARM,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)

REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.003,
    commission_per_share=0.000001,
    slippage_cents=0.05,
    slippage_proportional=0.0005,
    warmup_periods=WARM,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)

def test_cost_configuration(config, config_label):
    """Test momentum with a specific cost configuration."""
    print(f"\n=== Testing with {config_label} ===")
    print(f"  commission_per_trade: {config.commission_per_trade}")
    print(f"  commission_per_share: {config.commission_per_share}")
    print(f"  slippage_cents: {config.slippage_cents}")
    print(f"  slippage_proportional: {config.slippage_proportional}")
    
    manifest = load_manifest()
    tickers = {}
    
    # Load all tickers
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
    
    # Test each ticker
    for ticker in manifest["universe"]:
        bars = tickers[ticker]
        closes = bars.closes_array()
        
        # Generate momentum signals
        signals = bt.momentum_signals(closes, lookback=LOOKBACK)
        
        # Run walk_forward
        res = bt.walk_forward(
            list(bars), signals,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=config)
        
        # Calculate median log return
        log_returns = np.array([
            np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
            for f in res.folds], dtype=np.float64)
        
        if len(log_returns) > 0:
            median_return = float(np.median(log_returns))
            print(f"  {ticker}: median={median_return:+.3f}, n_folds={len(log_returns)}")

def main():
    print("=== Verifying Momentum with Explicit Cost Parameters ===")
    print(f"Dataset: {DATASET_ID}")
    
    test_cost_configuration(NO_COSTS, "NO COSTS")
    test_cost_configuration(REALISTIC_COSTS, "REALISTIC COSTS")
    
    print(f"\n=== Comparison ===")
    print("If costs are being applied, we should see different results")
    print("between NO COSTS and REALISTIC COSTS configurations.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
