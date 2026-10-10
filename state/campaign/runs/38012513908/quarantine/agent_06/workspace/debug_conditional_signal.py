"""Debug conditional_signal wrapper"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt
from research.backtest import momentum_signals, conditional_signal, load_ticker

# Test the conditional_signal wrapper
bars, dates = load_ticker('AAPL')
closes = bars.closes_array()

print(f'Closes length: {len(closes)}')
print(f'First 10 closes: {closes[:10]}')

# Create a simple test function that prints what's happening
def test_condition(early_vol):
    result = early_vol <= 0.25
    print(f'Vol: {early_vol:.3f}, Condition: {result}')
    return result

# Create wrapper
wrapped = conditional_signal(momentum_signals, test_condition)

# Test with different lookback values
print('Testing with lookback=1:')
signals1 = wrapped(closes, lookback=1)
print(f'Non-neutral signals: {sum(1 for s in signals1 if abs(s.weight) > 1e-12)}')

print('Testing with lookback=5:')
signals5 = wrapped(closes, lookback=5)
print(f'Non-neutral signals: {sum(1 for s in signals5 if abs(s.weight) > 1e-12)}')