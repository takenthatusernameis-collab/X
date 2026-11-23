"""Diagnostic: momentum sign convention, flips, and cost plumbing."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
bar_list = list(bars)

for lb in (3, 5, 10):
    sig = bt.momentum_signals(closes, lookback=lb)
    flips = sum(1 for i in range(1, len(sig)) if sig[i].weight != sig[i - 1].weight)
    nonneutral = sum(1 for s in sig if s.weight != 0.0)
    print(f"lookback {lb}: nonneutral={nonneutral} flips={flips}")

print()
print("sample closes[-8:]:", np.round(closes[-8:], 2).tolist())
sig5 = bt.momentum_signals(closes, lookback=5)
print("sample weights[-8:]:", [int(s.weight) for s in sig5[-8:]])

# What a sign-of-return momentum would look like
alt = []
for i in range(5, len(closes)):
    ret = closes[i] / closes[i - 5] - 1.0
    alt.append(1.0 if ret > 0 else -1.0)
flips_alt = sum(1 for i in range(1, len(alt)) if alt[i] != alt[i - 1])
print("alt (close[i]/close[i-5]-1) flips:", flips_alt)

for label, kw in [
    ("zero", {}),
    ("realistic", dict(commission_per_trade=2.0, commission_per_share=0.003,
                       slippage_cents=2.0, slippage_proportional=0.0005)),
]:
    cfg = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0, **kw)
    res = bt.walk_forward(bar_list, sig5, train_window=252, test_window=84,
                          warmup=60, overlap_window=60, cfg=cfg)
    tots = [round(f.metrics["total_return"], 6) for f in res.folds[:5]]
    print(label, "fold0..4 total_return:", tots, "n_folds:", len(res.folds))

for label, kw in [
    ("zero", {}),
    ("realistic", dict(commission_per_trade=2.0, commission_per_share=0.003,
                       slippage_cents=2.0, slippage_proportional=0.0005)),
]:
    cfg = bt.BacktestConfig(initial_capital=1e6, **kw)
    r = bt.run_bars(bar_list, sig5, cfg)
    print(label, "fills:", len(r.trades), "final equity:",
          round(float(r.equity_curve[-1]), 2),
          "total commission:", round(sum(f.commission for f in r.trades), 2))