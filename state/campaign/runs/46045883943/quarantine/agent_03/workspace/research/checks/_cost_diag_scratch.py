"""Tooling diagnostic 3: verify the notional-proportional commission is charged
as expected (commission magnitude, notional traded, fills). Mechanics only."""
import sys

import numpy as np

sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt

ticker = "AMZN"
bars, dates = bt.load_ticker(ticker)
closes = bars.closes_array()
ref_price = float(np.median(closes))
sig = bt.momentum_signals(closes, lookback=5)

for bps in (0, 30):
    cps = (bps / 10000.0) * ref_price
    cfg = bt.BacktestConfig(commission_per_share=cps)
    res = bt.walk_forward(list(bars), sig, train_window=252, test_window=84,
                          warmup=60, overlap_window=60, cfg=cfg)
    # aggregate over folds: total commission and notional in OOS windows
    tot_comm = 0.0
    tot_notional = 0.0
    for f in res.folds:
        # recompute fills not stored per fold; use full-run proxy instead
        pass
    # full-sample single run for fill inspection
    full = bt.run_bars(list(bars), sig, cfg)
    comm = sum(f.commission for f in full.trades)
    notional = sum(abs(f.shares) * f.price for f in full.trades)
    print(f"bps={bps:2d} fills={len(full.trades)} total_comm={comm:.1f} "
          f"total_notional={notional:.3e} comm/capital={comm/1e6:.4f} "
          f"notional/capital={notional/1e6:.2f}")
    med = float(np.median([np.log1p(np.clip(f.metrics['total_return'], -1+1e-12, None))
                           for f in res.folds]))
    print(f"     median_fold_log={med:+.4f}")

# how many flips (big deltas) vs small rebalances
full0 = bt.run_bars(list(bars), sig, bt.BacktestConfig())
deltas = [abs(f.shares) for f in full0.trades]
deltas = np.array(deltas)
print("fills:", len(deltas), "median|d|shares:", float(np.median(deltas)),
      "max|d|:", float(deltas.max()))
print("notional per fill: median", float(np.median(deltas))*ref_price,
      "max", float(deltas.max())*ref_price)