import numpy as np
import sys
sys.path.insert(0, '.')
import research.backtest as bt

closes = [100.0, 99.0]
signals = bt.momentum_signals(closes, lookback=1)
print('Actual momentum_signals:')
for i, s in enumerate(signals):
    print(f'  signals[{i}].weight = {s.weight}')

# Also test with positive returns
closes2 = [100.0, 101.0, 102.0]
signals2 = bt.momentum_signals(closes2, lookback=2)
print('\nActual momentum_signals positive:')
for i, s in enumerate(signals2):
    print(f'  signals[{i}].weight = {s.weight}')

# Test the calculation manually
print('\nManual calculation:')
lookback = 1
n = len(closes)
print(f'n={n}, lookback={lookback}')
print(f'range(lookback, n) = {list(range(lookback, n))}')
for i in range(lookback, n):
    ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
    print(f'i={i}, closes[{i-lookback+1}:{i+1}] = {closes[i-lookback+1:i+1]}')
    print(f'  ret = {ret}')
    print(f'  weight = {1.0 if ret > 0 else -1.0}')

print('\nNow test with positive:')
lookback = 2
n = len(closes2)
print(f'n={n}, lookback={lookback}')
print(f'range(lookback, n) = {list(range(lookback, n))}')
for i in range(lookback, n):
    ret = float(np.mean(np.log(closes2[i - lookback + 1 : i + 1])))
    print(f'i={i}, closes2[{i-lookback+1}:{i+1}] = {closes2[i-lookback+1:i+1]}')
    print(f'  ret = {ret}')
    print(f'  weight = {1.0 if ret > 0 else -1.0}')