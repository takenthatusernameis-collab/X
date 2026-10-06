"""Diagnostic 6: print signal weights vs trades side by side for the first
30 segment bars, NO and SEV, to see whether weights are +/-1 and how fills
relate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

bars, dates = bt.load_ticker("AAPL")
closes_full = bars.closes_array()
sig = bt.momentum_signals(closes_full, lookback=5)
seg_bars = list(bars)[60:90]   # 30 bars
seg_signals = sig[60:90]

print("bar(date)  close      sigwt   NO_shares  SEV_shares  NO_fill    SEV_fill")
NO = bt.BacktestConfig()
SEV = bt.BacktestConfig(slippage_cents=2.0, slippage_proportional=0.005)
pos0, possev = 0.0, 0.0
for k, bar in enumerate(seg_bars):
    sw = seg_signals[k]
    # recompute engine step for NO
    target0 = sw.weight * 1.0 * 1e6 / bar.close
    delta0 = target0 - pos0
    # SEV
    fillsev = bar.close + 2.0/100 + bar.close*0.005
    targetsev = sw.weight * 1e6 / bar.close  # shares sized on close (engine does this)
    # engine: target_shares uses bar.close; fill at close+slippage
    delsev = targetsev - possev
    print(f"{k+61:9d} {bar.close:9.4f} {sw.weight:+6.2f}  "
          f"pos0={pos0:10.1f} possev={possev:10.1f} "
          f"d0={delta0:+9.1f} dsev={delsev:+9.1f}")
    if abs(delsev) > 1e-9:
        cash0 = 0.0  # just tracking deltas; real engine carries cash
    if abs(delta0) > 1e-9:
        pass
    pos0, possev = target0, targetsev

# Now actual engine traces
tr0 = bt.run_bars(seg_bars, seg_signals, NO).trades
trsev = bt.run_bars(seg_bars, seg_signals, SEV).trades
print("\nEngine trades (first 12, SEV):")
for f in trsev[:12]:
    bar = seg_bars[f.date - 61]
    print(f"  date={f.date} shares={f.shares:+10.1f} fill={f.price:.4f} close={bar.close:.4f} "
          f"fill-close={f.price-bar.close:+.4f}")

# equity curves
eq0 = bt.run_bars(seg_bars, seg_signals, NO).equity_curve
eqsev = bt.run_bars(seg_bars, seg_signals, SEV).equity_curve
print(f"\nNO  equity[:6] = {[round(float(x),1) for x in eq0[:6]]}")
print(f"SEV equity[:6] = {[round(float(x),1) for x in eqsev[:6]]}")
print(f"SEV <= NO everywhere: {np.all(eqsev <= eq0 + 1e-6)}")
print(f"diff NO-SEV[:6] = {[round(float(a-b),1) for a,b in zip(eq0,eqsev)][:6]}")
