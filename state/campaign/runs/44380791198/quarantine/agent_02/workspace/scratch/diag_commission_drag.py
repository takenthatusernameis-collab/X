"""Scratch diagnostic: commission-only cost drag across the universe.

Since proportional slippage is structurally neutral for a continuously-invested
long/short flip strategy (see diag_slippage_asymmetry.py), the meaningful cost
stress is per-trade commission (and per-share commission). This script measures
the commission-only drag across all 10 assets at lookback 5, to calibrate a
commission-dominant cost grid that actually bites.

Also tests a per-share commission level ($0.005/share = half a cent per share),
which is a realistic institutional per-share charge and scales with activity.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

CAPITAL = 1e6
manifest = load_manifest()

# Cost levels to test (commission_per_trade, commission_per_share):
# A realistic institutional all-in cost for a liquid large-cap is ~1-5 bps of
# notional per round trip. At $1M notional, 1 bp = $100/round-trip = $50/trade.
# So $50/trade is ~1 bp round-trip. We test a grid that spans 0 to ~10 bps.
levels = [
    ("zero", 0.0, 0.0),
    ("comm_10", 10.0, 0.0),
    ("comm_50", 50.0, 0.0),       # ~1 bp round-trip at $1M
    ("comm_100", 100.0, 0.0),     # ~2 bp round-trip
    ("comm_500", 500.0, 0.0),     # ~10 bp round-trip (conservative)
]

print("=== Commission-only drag at lookback 5 (per-asset total return) ===")
print("  (proportional slippage is structurally neutral for this signal; see")
print("   diag_slippage_asymmetry.py -- only fixed commission bites)")
print("")
for entry in manifest["entries"]:
    t = entry["ticker"]
    bars, _ = bt.load_ticker(t)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    rets = {}
    for label, comm, comm_share in levels:
        cfg = bt.BacktestConfig(initial_capital=CAPITAL, commission_per_trade=comm,
                                commission_per_share=comm_share, slippage_cents=0.0)
        res = bt.run_bars(list(bars), sig, cfg)
        rets[label] = (res.equity_curve[-1] - CAPITAL) / CAPITAL
    base = rets["zero"]
    print(f"  {t:6s}: " + "  ".join(
        f"{label}={rets[label]:+.3f}({rets[label]-base:+.3f})" for label, _, _ in levels))
print("DIAG4 OK")