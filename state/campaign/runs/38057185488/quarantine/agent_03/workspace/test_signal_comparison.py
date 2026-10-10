#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/runner/work/X/X')
import research.backtest as bt
import numpy as np

# Load a ticker to test
bars, dates = bt.load_ticker('AAPL')
closes = bars.closes_array()

# Test ma_crossover_signals from the framework
fast, slow = 20, 60
result = bt.ma_crossover_signals(closes, fast, slow)
print(f'Framework ma_crossover_signals result: {result[0].date}, {result[0].weight}')

# Test what the verification script's base_ma_signals would do
n = len(closes)
out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
fast, slow = int(fast), int(slow)
for i in range(max(fast, slow) - 1, n):
    fast_ma = float(np.mean(closes[i - fast + 1 : i + 1]))
    slow_ma = float(np.mean(closes[i - slow + 1 : i + 1]))
    out[i] = bt.Signal(date=i + 1, weight=1.0 if fast_ma > slow_ma else -1.0)
    
print(f'Verification script base_ma_signals result: {out[0].date}, {out[0].weight}')
print(f'Are they the same? {result == out}')