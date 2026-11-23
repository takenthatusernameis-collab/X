"""Diagnostic: momentum signal look-ahead and degeneracy, with a causal fix.

Two defects in the existing `research.backtest.momentum_signals`:

1. Look-ahead: the log-return expression `np.mean(np.log(closes[i-lookback+1:i+1]))`
   is the mean LOG PRICE over the window (a monotone transform of the window's
   mean price), not a return. Its sign compares the mean price over the trailing
   window with zero, so the signal compares current prices with a fixed zero
   level. Because the whole collected universe is far above zero for the entire
   2009-2026 sample, the signal is ~constant +1, flips once, and is effectively
   "always long" -- a buy-and-hold bet, not a momentum bet. It also carries a
   textbook look-ahead (uses close[i] to set the position held into i+1).

2. The diagnostic below measures both the defect and a causal candidate fix.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def momentum_causal(closes, lookback=5):
    """Past-only momentum: long/short the sign of the previous lookback-day return.

    Position held into bar i+1 is set from closes[:i+1] only, i.e. the signal
    at bar i uses closes[i] (the close the position was entered at) and the
    trailing return closes[i]/closes[i-lookback] - 1.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(closes[i] / closes[i - lookback] - 1.0)
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()

for lb in (3, 5, 10):
    cur = bt.momentum_signals(closes, lookback=lb)
    new = momentum_causal(closes, lookback=lb)
    cur_flips = sum(1 for i in range(1, len(cur)) if cur[i].weight != cur[i - 1].weight)
    new_flips = sum(1 for i in range(1, len(new)) if new[i].weight != new[i - 1].weight)
    print(f"lookback {lb}: current flips={cur_flips} (nonneutral="
          f"{sum(1 for s in cur if s.weight != 0)}) | causal flips={new_flips}")

print()
print("current weights[-8:]", [int(s.weight) for s in bt.momentum_signals(closes, lookback=5)[-8:]])
print("causal  weights[-8:]", [int(s.weight) for s in momentum_causal(closes, lookback=5)[-8:]])

print()
for label, sigfn in [("current", bt.momentum_signals), ("causal", momentum_causal)]:
    for cost_name, kw in [
        ("zero", {}),
        ("realistic", dict(commission_per_trade=2.0, commission_per_share=0.003,
                           slippage_cents=2.0, slippage_proportional=0.0005)),
        ("stress2x", dict(commission_per_trade=4.0, commission_per_share=0.006,
                          slippage_cents=4.0, slippage_proportional=0.0010)),
    ]:
        cfg = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0, **kw)
        r = bt.run_bars(list(bars), sigfn(closes, lookback=5), cfg)
        tot = (r.equity_curve[-1] - 1e6) / 1e6
        print(f"{label:8s} {cost_name:10s} fills={len(r.trades):5d} "
              f"total_return={tot:+.3f} commission={sum(f.commission for f in r.trades):.0f}")