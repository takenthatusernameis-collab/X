#!/usr/bin/env python3
"""Simple test of momentum_signals function."""
import sys
sys.path.insert(0, '.')
import research.backtest as bt
import numpy as np

print('Testing momentum_signals...')
closes = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
signals = bt.momentum_signals(closes, lookback=3)
print('Signals created:', len(signals))
print('First signal weight:', signals[0].weight)
print('Testing passed!')