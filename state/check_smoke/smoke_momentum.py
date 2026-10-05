"""Synthetic smoke test for the momentum signal class."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt

np.random.seed(0)
closes = np.cumsum(np.random.randn(500) * 0.02) + 100.0
sig = bt.momentum_signals(closes, lookback=5)
assert len(sig) == len(closes)
assert all(s.weight == 0.0 for s in sig[:5]), "should be neutral before lookback"
assert all(s.weight in (-1.0, 1.0) for s in sig[5:]), "should be +/-1 after lookback"
mrev = bt.mean_reversion_signals(closes, lookback=5)
for a, b in zip(sig[5:], mrev[5:]):
    assert a.weight == -b.weight, "momentum and reversal must be opposite sign"

bars = list(bt.generate_bars(n_bars=500, seed=0))
res = bt.run_bars(bars, bt.always_long_signal(closes, 5, 5), bt.BacktestConfig(initial_capital=1e6, periods_per_year=252))
print("signal contract OK; opposite-of-reversal OK; deterministic edge earns {:+.3f}".format(
    (res.equity_curve[-1] - 1e6) / 1e6))
print("SMOKE TEST PASSED")
