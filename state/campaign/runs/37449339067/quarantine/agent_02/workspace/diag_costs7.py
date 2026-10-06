"""Diagnostic 7: find where SEV equity crosses above NO on the full series."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

bars, dates = bt.load_ticker("AAPL")
closes_full = bars.closes_array()
sig = bt.momentum_signals(closes_full, lookback=5)

NO = bt.BacktestConfig()
SEV = bt.BacktestConfig(slippage_cents=2.0, slippage_proportional=0.005)

f0 = bt.run_bars(list(bars), sig, NO)
fs = bt.run_bars(list(bars), sig, SEV)

diff = f0.equity_curve - fs.equity_curve
first_cross = np.where(diff < -1e-6)[0]
print(f"First bar where SEV > NO (diff<0): index {first_cross[0]} (date {first_cross[0]+1}), "
      f"diff={diff[first_cross[0]]:+.2f}")
print(f"Bars with SEV>NO: {len(first_cross)} out of {len(diff)}")
print(f"Final diff (NO-SEV): {diff[-1]:+.2f}")


def trace_until(cfg, up_to):
    cash = 1e6
    shares = 0.0
    out = []
    for k, bar in enumerate(bars):
        if k > up_to:
            break
        signal = sig[k]
        target = signal.weight * 1.0 * 1e6 / bar.close
        delta = target - shares
        fill = bar.close + cfg.slippage_cents/100 + bar.close*cfg.slippage_proportional
        commission = 0.0
        if abs(delta) > 1e-12:
            if delta > 0:
                cash -= delta*fill + commission
            else:
                cash += -delta*fill - commission
            shares = target
        eq = cash + shares*bar.close
        out.append((bar.date, bar.close, cash, shares, eq))
    return out


i0 = first_cross[0]
print(f"\n=== Trace around crossing, bars {i0-6}..{i0+6} ===")
print("  bar   close       cash      shares     equity  (NO / SEV)")
t0 = trace_until(NO, i0+6)
tsev = trace_until(SEV, i0+6)
for (_, c0, co0, so0, eo0), (_, cs, css, ssv, esv) in zip(t0, tsev):
    print(f"  {c0:9.4f}  {co0:12.1f}/{cs:12.1f}  {so0:10.1f}/{ssv:10.1f}  "
          f"{eo0:12.1f}/{esv:12.1f}")
