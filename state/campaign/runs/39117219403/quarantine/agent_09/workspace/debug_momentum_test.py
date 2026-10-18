#!/usr/bin/env python3
import numpy as np
from research.backtest.regime_stability import momentum_volatility_normalized_signals

# Test the exact case from the failing test
closes_down = np.array([110.0, 109.0, 108.0, 107.0, 106.0])
print('Test case: closes_down =', closes_down)
print('lookback: 3')

# Get the signals
signals_down = momentum_volatility_normalized_signals(closes_down, lookback=3)
print('Signal weights:', [round(s.weight, 6) for s in signals_down])

# Check the signal at index 3 (bar 4)
print('\nSignal at index 3 (bar 4):')
print('  weight =', signals_down[3].weight)
print('  date =', signals_down[3].date)

# Manual calculation for signal at index 3
# The signal at index 3 uses closes_down[1:4] = [109.0, 108.0, 107.0]
print('\nManual calculation for signal at index 3:')
print('  closes used:', closes_down[1:4])
print('  log(109.0/108.0) =', np.log(109.0/108.0))
print('  log(108.0/107.0) =', np.log(108.0/107.0))
ret = float(np.mean(np.log(closes_down[1:4])))
print('  mean log return =', ret)
print('  signal_weight =', 1.0 if ret > 0 else -1.0)
print('  With vol_scale_factor=1.0 and capped at 1.0, weight should be:', 1.0 if ret > 0 else -1.0)

# Test with a case that should produce a negative weight
print('\n\nTesting with a case that should produce negative weight:')
closes_neg = np.array([100.0, 99.0, 98.0, 97.0, 96.0])
print('closes_neg =', closes_neg)
signals_neg = momentum_volatility_normalized_signals(closes_neg, lookback=3)
print('Signal weights:', [round(s.weight, 6) for s in signals_neg])
print('Signal at index 3 weight =', signals_neg[3].weight)

# Manual calculation
ret_neg = float(np.mean(np.log(closes_neg[1:4])))
print('  mean log return =', ret_neg)
print('  signal_weight =', 1.0 if ret_neg > 0 else -1.0)

# Test with the original data from momentum_volatility_normalized.py
print('\n\nTesting with original momentum_volatility_normalized.py data:')
closes_orig = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0])
print('closes_orig =', closes_orig)
signals_orig = momentum_volatility_normalized_signals(closes_orig, lookback=3)
print('Signal at index 3 weight =', signals_orig[3].weight)
ret_orig = float(np.mean(np.log(closes_orig[1:4])))
print('  mean log return =', ret_orig)
print('  signal_weight =', 1.0 if ret_orig > 0 else -1.0)

print('\nConclusion: The issue is that all test cases with decreasing prices still produce positive mean log returns because the log calculation is on ratios, not absolute prices.')
