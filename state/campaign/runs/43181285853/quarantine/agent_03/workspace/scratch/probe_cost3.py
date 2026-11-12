import sys
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
runs = []
cur = labels[0]; start = 0
for i in range(1, len(labels)):
    if labels[i] != cur:
        if i - start >= 400:
            runs.append((cur, start, i))
        cur = labels[i]; start = i
if len(labels) - start >= 400:
    runs.append((cur, start, len(labels)))
print("runs", runs)
name, s, e = runs[0]
seg_bars = list(bars)[s:e]
sig = bt.momentum_signals(closes, lookback=5)[s:e]

for cname, cfg in [
    ("zero", bt.BacktestConfig()),
    ("realistic", bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                                    slippage_cents=2.0, slippage_proportional=0.0005)),
]:
    res = bt.walk_forward(seg_bars, sig, train_window=252, test_window=84, warmup=60,
                          overlap_window=60, cfg=cfg)
    comms = []
    for f in res.folds:
        # recompute fills for this fold by rerunning the fold window is hard; use metrics
        pass
    print(cname, "n_folds", len(res.folds),
          "median_log", round(res.aggregate_metrics["median_log_total_return"], 6),
          "mean_log", round(res.aggregate_metrics["mean_log_total_return"], 6))

# Direct single fold: measure total commission paid
from research.backtest.engine import Backtest
fold_start = 0
fold_end = 60 + 252 + 84
fb = list(bars)[s:s+fold_end]
fs = sig[:fold_end]
for cname, cfg in [
    ("zero", bt.BacktestConfig()),
    ("realistic", bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                                    slippage_cents=2.0, slippage_proportional=0.0005)),
]:
    fc = bt.BacktestConfig(initial_capital=cfg.initial_capital, target_exposure=cfg.target_exposure,
                           commission_per_trade=cfg.commission_per_trade,
                           commission_per_share=cfg.commission_per_share,
                           slippage_cents=cfg.slippage_cents,
                           slippage_proportional=cfg.slippage_proportional,
                           warmup_periods=60)
    r = Backtest(fc).run(fb, fs)
    tot_comm = sum(f.commission for f in r.trades)
    gross = sum(abs(f.shares) * f.price for f in r.trades)
    test_eq = r.equity_curve[-84:]
    print(cname, "fills", len(r.trades), "commissions", round(tot_comm, 2),
          "gross_traded", round(gross, 1), "test_first", round(test_eq[0], 2),
          "test_last", round(test_eq[-1], 2), "test_ret", round(test_eq[-1]/test_eq[0]-1, 6))