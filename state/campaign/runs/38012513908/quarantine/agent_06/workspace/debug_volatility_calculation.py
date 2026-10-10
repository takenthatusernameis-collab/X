"""Debug volatility calculation in conditional_signal"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

# Test volatility calculation
bars, dates = bt.load_ticker('AAPL')
closes = bars.closes_array()

print(f'Closes length: {len(closes)}')

# Test the conditional_signal logic directly
n = len(closes)
window = min(60, max(4, n // 4))
print(f'Window: {window}')
print(f'Window//2: {window//2}')
print(f'Window: {window}')

early_vol = float(np.std(np.log(closes[window // 2 : window]), ddof=1)) * np.sqrt(252.0)
print(f'early_vol: {early_vol:.3f}')
print(f'early_vol > 0.25: {early_vol > 0.25}')
print(f'Condition (early_vol <= 0.25): {early_vol <= 0.25}')

# Test with different windows
for w in [20, 40, 60, 80]:
    if w <= len(closes):
        early_vol = float(np.std(np.log(closes[w//2:w]), ddof=1)) * np.sqrt(252.0)
        print(f'Window={w}: early_vol={early_vol:.3f}, Condition={early_vol <= 0.25}')