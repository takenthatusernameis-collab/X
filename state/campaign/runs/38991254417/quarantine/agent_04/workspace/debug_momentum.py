#!/usr/bin/env python3
"""
Debug momentum signal behavior.
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
print("\nPrice changes:", closes[1:] - closes[:-1])

# Test the simple momentum signal
print("\n=== Simple momentum signal analysis ===")
for i in range(5, 20):
    # Calculate lookback return manually
    lookback = 5
    start_idx = i - lookback + 1
    if start_idx >= 0:
        lookback_return = float(np.mean(np.log(closes[start_idx:i+1])))
        print(f"Bar {i+1}: lookback return = {lookback_return:.4f}, signal = {'LONG' if lookback_return > 0 else 'SHORT'}")

# Generate the signals
base_signals = bt.momentum_signals(closes, lookback=5)
print("\n=== Base momentum signals (bars 5-20) ===")
for i in range(5, 20):
    print(f"Bar {i+1}: weight={base_signals[i].weight}")