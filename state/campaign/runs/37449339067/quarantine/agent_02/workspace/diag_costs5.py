"""Diagnostic 5: FULL series (dates start at 1, so equity-fill check is valid)
NO vs SEV; exact slippage cost accounting; walk-forward fold-by-fold diff."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

bars, dates = bt.load_ticker("AAPL")
closes_full = bars.closes_array()
labels = bt.volatility_blocks(closes_full, n_blocks=4, window=60)
sig = bt.momentum_signals(closes_full, lookback=5)

NO = bt.BacktestConfig()
SEV = bt.BacktestConfig(slippage_cents=2.0, slippage_proportional=0.005)

f0 = bt.run_bars(list(bars), sig, NO)
fs = bt.run_bars(list(bars), sig, SEV)
bt.check_equity_matches_fills(f0.equity_curve, f0.trades, closes_full, 1e6)
bt.check_equity_matches_fills(fs.equity_curve, fs.trades, closes_full, 1e6)
print("Full-series equity-fill check: PASS")

sev_cost_full = sum(
    (f.price - closes_full[f.date - 1]) * abs(f.shares) for f in fs.trades)
print(f"\nFull series: NO final={f0.equity_curve[-1]:.2f}, SEV final={fs.equity_curve[-1]:.2f}, "
      f"gap={f0.equity_curve[-1]-fs.equity_curve[-1]:+.2f}")
print(f"Full series SEV slippage cost: {sev_cost_full:.2f}")
print(f"SEV final <= NO final: {fs.equity_curve[-1] <= f0.equity_curve[-1]} "
      f"(with {sev_cost_full:.2f} slippage paid)")

# Walk-forward on the FULL series (actual framework usage)
def fold_returns(cfg):
    res = bt.walk_forward(list(bars), sig, train_window=252, test_window=84,
                          warmup=60, overlap_window=60, cfg=cfg)
    tr = [float(f.metrics["total_return"]) for f in res.folds]
    return tr, res

no_tr, no_res = fold_returns(NO)
sev_tr, sev_res = fold_returns(SEV)
print(f"\nWFO folds: NO {len(no_res.folds)} == SEV {len(sev_res.folds)}")
no_log = [np.log1p(x) for x in no_tr]
sev_log = [np.log1p(x) for x in sev_tr]
print(f"NO  median_log={np.median(no_log):+.6f} mean={np.mean(no_tr):+.6f}")
print(f"SEV median_log={np.median(sev_log):+.6f} mean={np.mean(sev_tr):+.6f}")
diffs = [a - b for a, b in zip(no_tr, sev_tr)]
print(f"fold NO-SEV diff: min={np.min(diffs):+.6f}, max={np.max(diffs):+.6f}, "
      f"median={np.median(diffs):+.6f}, all>=0: {all(d >= -1e-12 for d in diffs)}")
bad = [(i, a, b) for i, (a, b) in enumerate(zip(no_tr, sev_tr)) if a - b < -1e-9]
print(f"folds where SEV > NO: {len(bad)}")
for i, a, b in bad[:8]:
    print(f"   fold {i}: NO={a:+.6f} SEV={b:+.6f} diff={a-b:+.6f}")
