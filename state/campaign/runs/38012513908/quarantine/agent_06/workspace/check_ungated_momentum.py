"""Check ungated momentum behavior for comparison"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt
from research.backtest import momentum_signals, load_ticker

# Test ungated momentum
bars, dates = load_ticker('AAPL')
closes = bars.closes_array()

print(f'Closes length: {len(closes)}')

# Generate ungated momentum signals
signals = momentum_signals(closes, lookback=5)

nonneutral = sum((1 for s in signals if abs(s.weight) > 1e-12))
long_positions = sum((1 for s in signals if s.weight > 0.5))
short_positions = sum((1 for s in signals if s.weight < -0.5))

print(f'\nUNGATED MOMENTUM (lookback=5)')
print(f'Total signals: {len(signals)}')
print(f'Non-neutral signals: {nonneutral}')
print(f'Long positions: {long_positions}')
print(f'Short positions: {short_positions}')

# Check some signal values
print(f'\nSample signal values:')
for i in range(10, min(30, len(signals))):
    if abs(signals[i].weight) > 1e-12:
        print(f'  Bar {signals[i].date}: weight={signals[i].weight:.3f}')

# Calculate the momentum returns for these signals
print(f'\nMomentum returns for non-neutral signals:')
for i in range(10, min(30, len(signals))):
    if abs(signals[i].weight) > 1e-12:
        ret = np.mean(np.log(closes[i-4:i+1]))
        print(f'  Bar {signals[i].date}: momentum={ret:.3f}, signal={signals[i].weight:.3f}')