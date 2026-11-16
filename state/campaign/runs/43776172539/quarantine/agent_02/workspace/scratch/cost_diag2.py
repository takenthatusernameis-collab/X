import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)
sig = bt.momentum_signals(closes, lookback=5)

# Use the first turbulent segment [60,1116) as in the diagnostic.
seg_bars = list(bars)[60:1116]
seg_sig = sig[60:1116]

zero = bt.BacktestConfig(initial_capital=1e6)
real = bt.BacktestConfig(initial_capital=1e6, commission_per_trade=2.0,
                         commission_per_share=0.003, slippage_cents=2.0,
                         slippage_proportional=0.0005)
cons = bt.BacktestConfig(initial_capital=1e6, commission_per_trade=4.0,
                         commission_per_share=0.006, slippage_cents=4.0,
                         slippage_proportional=0.0010)

for cname, cfg in [("zero", zero), ("realistic", real), ("conservative", cons)]:
    wf = bt.walk_forward(seg_bars, seg_sig, train_window=252, test_window=84,
                         warmup=60, overlap_window=60, cfg=cfg)
    rets = np.array([f.metrics["total_return"] for f in wf.folds])
    logs = np.log1p(np.clip(rets, -1 + 1e-12, None))
    print(f"{cname:12s} n_folds={len(rets)} mean_log={logs.mean():.8f} "
          f"median_log={np.median(logs):.8f} mean_ret={rets.mean():.8f} "
          f"median_ret={np.median(rets):.8f}")

# Also measure turnover: total abs traded notional / initial capital.
for cname, cfg in [("zero", zero), ("realistic", real), ("conservative", cons)]:
    r = bt.run_bars(seg_bars, seg_sig, cfg)
    gross = sum(abs(f.shares) * f.price for f in r.trades)
    print(f"{cname:12s} fills={len(r.trades)} gross_traded={gross:.0f} "
          f"gross/capital={gross/1e6:.2f}")