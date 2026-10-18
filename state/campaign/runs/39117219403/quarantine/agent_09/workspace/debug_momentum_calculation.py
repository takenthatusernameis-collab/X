#!/usr/bin/env python3
import numpy as np
from research.backtest.regime_stability import momentum_volatility_normalized_signals

# Test with the new test data
closes_down = np.array([100.0, 98.0, 96.0, 94.0, 92.0])
print('Test data: closes_down =', closes_down)
print('lookback: 3')

# Get the signals
signals_down = momentum_volatility_normalized_signals(closes_down, lookback=3)
print('Signal weights:', [round(s.weight, 6) for s in signals_down])

# Check the signal at index 3 (bar 4)
print('\nSignal at index 3 (bar 4):')
print('  weight =', signals_down[3].weight)
print('  date =', signals_down[3].date)

# Manual calculation for signal at index 3
# The signal at index 3 uses closes_down[1:4] = [98.0, 96.0, 94.0]
print('\nManual calculation for signal at index 3:')
print('  closes used:', closes_down[1:4])
print('  log(98.0/96.0) =', np.log(98.0/96.0))
print('  log(96.0/94.0) =', np.log(96.0/94.0))
ret = float(np.mean(np.log(closes_down[1:4])))
print('  mean log return =', ret)
print('  signal_weight =', 1.0 if ret > 0 else -1.0)
print('  With vol_scale_factor=1.0 and capped at 1.0, weight should be:', 1.0 if ret > 0 else -1.0)

# Let's understand what the log calculation is doing
print('\nUnderstanding the log calculation:')
print('For momentum calculation, we need to calculate log returns')
print('log_return_1 = log(price_t / price_{t-1})')
print('log_return_2 = log(price_{t-1} / price_{t-2})')
print('')
print('For our data:')
print('  price_1 = 98.0, price_0 = 100.0')
print('  price_2 = 96.0, price_1 = 98.0')
print('  price_3 = 94.0, price_2 = 96.0')
print('')
print('log(98.0/100.0) =', np.log(98.0/100.0))
print('log(96.0/98.0) =', np.log(96.0/98.0))
print('log(94.0/96.0) =', np.log(94.0/96.0))
print('')
print('Mean log return = (log(98.0/100.0) + log(96.0/98.0) + log(94.0/96.0)) / 3')
print('              = ({:.6f} + {:.6f} + {:.6f}) / 3'.format(np.log(98.0/100.0), np.log(96.0/98.0), np.log(94.0/96.0)))
print('              = {:.6f} / 3'.format(np.log(98.0/100.0) + np.log(96.0/98.0) + np.log(94.0/96.0)))
print('              = {:.6f}'.format(np.log(98.0/100.0) + np.log(96.0/98.0) + np.log(94.0/96.0)))

# Wait, I think I misunderstood the momentum calculation
print('\nActually, looking at the code in regime_stability.py:')
print('The momentum calculation uses: ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))')
print('For signal at index 3 (i=3) with lookback=3:')
print('  closes[i - lookback + 1 : i + 1] = closes[3 - 3 + 1 : 3 + 1] = closes[1:4]')
print('  So it uses closes[1], closes[2], closes[3] = [98.0, 96.0, 94.0]')
print('  This means it calculates: mean(log(98.0), log(96.0), log(94.0))')
print('  = (log(98.0) - log(100.0), log(96.0) - log(98.0), log(94.0) - log(96.0)) / 3')
print('')
print('Let me calculate this:')
print('  log(98.0) - log(100.0) =', np.log(98.0) - np.log(100.0))
print('  log(96.0) - log(98.0) =', np.log(96.0) - np.log(98.0))
print('  log(94.0) - log(96.0) =', np.log(94.0) - np.log(96.0))
mean_diff = (np.log(98.0) - np.log(100.0) + np.log(96.0) - np.log(98.0) + np.log(94.0) - np.log(96.0)) / 3
print('  Mean diff =', mean_diff)
print('  This is approximately:', mean_diff)
print('')
print('So the momentum calculation is using differences of log prices, not log returns!')
print('This means it calculates: mean(log(price_t) - log(price_{t-1})) for each t')
print('This is equivalent to: mean(log(price_t / price_{t-1})) - log(price_0 / price_0)')
print('Which simplifies to: mean(log(price_t / price_{t-1}))')
print('')
print('But wait, let me check the actual implementation again...')
print('')
print('Looking at the momentum_signals function in regime_stability.py:')
print('  ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))')
print('  out[i] = Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)')
print('')
print('This is confusing. Let me just run the actual momentum_signals function to see what it produces...')

# Import momentum_signals
from research.backtest.regime_stability import momentum_signals

print('\nTesting with momentum_signals (for comparison):')
signals_momentum = momentum_signals(closes_down, lookback=3)
print('Signal weights:', [round(s.weight, 6) for s in signals_momentum])
print('Signal at index 3 weight =', signals_momentum[3].weight)
ret_momentum = float(np.mean(np.log(closes_down[1:4])))
print('  mean log return =', ret_momentum)
print('  signal_weight =', 1.0 if ret_momentum > 0 else -1.0)
