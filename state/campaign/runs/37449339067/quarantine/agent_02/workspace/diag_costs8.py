"""Diagnostic 8: direct engine trace vs manual trace for first 12 bars,
showing signal weight, trade, cash, shares, equity for NO and SEV."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)
seg_bars = list(bars)[:12]
seg_signals = sig[:12]

NO = bt.BacktestConfig()
SEV = bt.BacktestConfig(slippage_cents=2.0, slippage_proportional=0.005)

def run(cfg, name):
    res = bt.run_bars(seg_bars, seg_signals, cfg)
    print(f"\n--- {name} ---")
    print("bar  close     wt   fill    shares  cash      equity     delta")
    cash = 1e6
    shares = 0.0
    for k, bar in enumerate(seg_bars):
        eq_before = cash + shares*bar.close
        signal = seg_signals[k]
        target = signal.weight * 1.0 * 1e6 / bar.close
        delta = target - shares
        fill = bar.close + cfg.slippage_cents/100 + bar.close*cfg.slippage_proportional
        if abs(delta) > 1e-9:
            if delta > 0:
                cash -= delta*fill
            else:
                cash += -delta*fill
            shares = target
        eq = cash + shares*bar.close
        e = res.equity_curve[k]
        flag = "MISMATCH" if abs(eq - e) > 1e-6 else ""
        print(f"{k:3d} {bar.close:8.4f} {signal.weight:+6.1f} {fill:7.4f} "
              f"{shares:10.1f} {cash:11.1f} {eq:12.1f} {e:12.1f} {flag}")
    return res

r0 = run(NO, "NO (no cost)")
rsev = run(SEV, "SEV (2c + 0.5%)")
