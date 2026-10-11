import numpy as np

# Direct implementation of momentum_signals to understand the behavior

def momentum_signals_debug(closes, lookback):
    """Debug version of momentum_signals"""
    n = len(closes)
    out = [(i + 1, 0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = (i + 1, 1.0 if ret > 0 else -1.0)
    return out

# Test cases
print("=== Test 1: Negative return ===")
closes = [100.0, 99.0]
result = momentum_signals_debug(closes, lookback=1)
print(f"closes: {closes}")
print(f"result: {result}")
print()

print("=== Test 2: Positive return ===")
closes = [100.0, 101.0, 102.0]
result = momentum_signals_debug(closes, lookback=2)
print(f"closes: {closes}")
print(f"result: {result}")
print()

print("=== Test 3: Three-element lookback ===")
closes = [100.0, 101.0, 102.0, 103.0]
result = momentum_signals_debug(closes, lookback=3)
print(f"closes: {closes}")
print(f"result: {result}")
print()

print("=== Test 4: All positive ===")
closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
result = momentum_signals_debug(closes, lookback=3)
print(f"closes: {closes}")
print(f"result: {result}")