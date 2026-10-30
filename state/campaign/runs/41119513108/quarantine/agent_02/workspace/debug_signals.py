"""Debug random signal generation for null benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

# Test random signal generation
seed = 42
rng = np.random.default_rng(seed)
weights = rng.choice([-1.0, 0.0, 1.0], size=10, p=[1.0/3.0, 1.0/3.0, 1.0/3.0])
print('Random weights:', weights)
print('Number of non-zero weights:', np.sum(np.abs(weights) > 0))
print('Number of zero weights:', np.sum(weights == 0))

# Test the hash-based seed generation
import hashlib

def generate_matched_null_signals_debug(closes, base_signals, seed=42, lookback=None):
    """Generate matched null signals with same timing as base signals.
    
    For each position in base_signals where weight != 0, assign a random sign
    (-1 or 1) with equal probability, preserving the timing of signals.
    Uses a unique seed per (base_signals, lookback) combination to avoid collisions.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    
    # Create unique seed for this (base_signals, lookback) combination
    # Hash the signals and lookback to get a unique but deterministic seed
    signals_hash = hash(tuple(sig.weight for sig in base_signals))
    combined_seed = hash((signals_hash, lookback, seed)) & 0x7FFFFFFF  # Ensure positive
    
    print(f"signals_hash: {signals_hash}, lookback: {lookback}, combined_seed: {combined_seed}")
    
    rng = np.random.default_rng(combined_seed)
    for i in range(n):
        if base_signals[i].weight != 0.0:
            # Assign random sign, preserving the timing
            out[i] = bt.Signal(date=i + 1, weight=float(rng.choice([-1.0, 1.0])))
    return out

# Test with synthetic data
bars = bt.generate_bars(100, seed=42)
closes = bars.closes_array()

# Test momentum signals for different lookbacks
for lookback in [3, 5, 10]:
    print(f"\n=== Lookback {lookback} ===")
    momentum_signals = bt.momentum_signals(closes, lookback=lookback)
    
    # Count non-zero signals
    non_zero_count = sum(1 for sig in momentum_signals if sig.weight != 0.0)
    print(f"Momentum signals non-zero count: {non_zero_count}")
    
    # Generate null signals
    null_signals = generate_matched_null_signals_debug(closes, momentum_signals, seed=seed, lookback=lookback)
    
    # Count non-zero null signals
    null_non_zero_count = sum(1 for sig in null_signals if sig.weight != 0.0)
    print(f"Null signals non-zero count: {null_non_zero_count}")
    
    # Show first 10 signals for comparison
    print("First 10 momentum signals:", [sig.weight for sig in momentum_signals[:10]])
    print("First 10 null signals:", [sig.weight for sig in null_signals[:10]])
