"""Scratch probe: confirm the realistic cost model actually reduces momentum
walk-forward returns, and measure turnover. Not a research result."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

REAL = dict(commission_per_trade=2.0, commission_per_share=0.003,
            slippage_cents=2.0, slippage_proportional=0.0005)

bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

# count flips in the signal
flips = sum(1 for i in range(1, len(sig)) if sig[i].weight != sig[i - 1].weight)
print(f"AAPL bars={len(closes)} signal flips={flips}")

for label, extra in (("zero", {}), ("realistic", REAL)):
    cfg = bt.BacktestConfig(initial_capital=1e6, **extra)
    res = bt.walk_forward(list(bars), sig, train_window=252, test_window=84,
                          warmup=60, overlap_window=60, cfg=cfg)
    m = np.array([f.metrics["total_return"] for f in res.folds])
    ml = np.log1p(np.clip(m, -1 + 1e-12, None))
    trades = sum(f.metrics["n_trades"] for f in res.folds)
    print(f"  {label:10s}: folds={res.aggregate_metrics['n_folds']} "
          f"mean_log={np.mean(ml):+.5f} median_log={np.median(ml):+.5f} "
          f"total_folds_trades={trades}")

# Single fold equity comparison
cfg0 = bt.BacktestConfig(initial_capital=1e6)
cfg1 = bt.BacktestConfig(initial_capital=1e6, **REAL)
r0 = bt.run_bars(list(bars), sig, cfg0)
r1 = bt.run_bars(list(bars), sig, cfg1)
print(f"full-sample fills: zero={len(r0.trades)} realistic={len(r1.trades)}")
print(f"full-sample total_return: zero={(r0.equity_curve[-1]-1e6)/1e6:+.4f} "
      f"realistic={(r1.equity_curve[-1]-1e6)/1e6:+.4f}")
comm = sum(f.commission for f in r1.trades)
print(f"realistic total commissions={comm:.2f} over {len(r1.trades)} fills")