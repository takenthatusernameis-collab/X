import sys
sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt
import numpy as np

bars, dates = bt.load_ticker("AAPL")
bars_list = list(bars)
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

c0 = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0, commission_per_share=0.0,
                       commission_per_trade=1.0, slippage_cents=0.0, slippage_proportional=0.0)
c1 = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0, commission_per_share=0.005,
                       commission_per_trade=1.0, slippage_cents=1.0, slippage_proportional=0.001)

wf0 = bt.walk_forward(bars_list[0:396], sig[0:396], train_window=252, test_window=84, warmup=60,
                      overlap_window=60, cfg=c0)
wf1 = bt.walk_forward(bars_list[0:396], sig[0:396], train_window=252, test_window=84, warmup=60,
                      overlap_window=60, cfg=c1)

print("walk_forward folds (first 3):")
for i in range(3):
    m0 = wf0.folds[i].metrics
    m1 = wf1.folds[i].metrics
    print(" fold {}: zero total_return={:.3f} n_trades={}; base total_return={:.3f} n_trades={}".format(
        i, m0["total_return"], m0.get("n_trades"), m1["total_return"], m1.get("n_trades")))
print(" zero  med:", [round(np.log1p(np.clip(f.metrics["total_return"], -1+1e-12, None))) for f in wf0.folds])
print(" base  med:", [round(np.log1p(np.clip(f.metrics["total_return"], -1+1e-12, None))) for f in wf1.folds])

# Check fold config actually has costs
import inspect
src = inspect.getsource(bt.walk_forward)
print("\nwalk_forward has fold_cfg with cost fields:",
      "commission_per_share=cfg.commission_per_share" in src and
      "slippage_proportional=cfg.slippage_proportional" in src)
