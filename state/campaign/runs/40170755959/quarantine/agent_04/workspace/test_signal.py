#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, 'research/backtest')
from research.backtest.regime_stability import stress_segments, segment_fn_from_labels, volatility_blocks
from research.backtest import momentum_signals, generate_bars

# Generate test data
bars = generate_bars(600, seed=42)
closes = bars.closes_array()

# Test run_asset function
lookback = 5
bars_obj = {'AAPL': bars}

print('bars type:', type(bars))
print('bars.closes_array() type:', type(bars.closes_array()))

# Try to simulate what run_asset does
labels = volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = segment_fn_from_labels(labels)
sig = momentum_signals(closes, lookback=lookback)

# Test the hold_effective_signals function
for hold in [1, 2, 3, 5]:
    eff = []
    for i in range(len(sig)):
        if i < lookback:
            eff.append(sig[0].__class__(date=i + 1, weight=0.0))
        else:
            idx = i - (i - lookback) % hold
            eff.append(sig[0].__class__(date=i + 1, weight=sig[idx].weight))
    
    print(f'hold={hold}: len(sig)={len(sig)}, len(eff)={len(eff)}')
    print(f'hold={hold}: sig[0]={type(sig[0])}, eff[0]={type(eff[0])}')
    print(f'hold={hold}: sig[0]={sig[0]}')
    print(f'hold={hold}: eff[0]={eff[0]}')
    print()