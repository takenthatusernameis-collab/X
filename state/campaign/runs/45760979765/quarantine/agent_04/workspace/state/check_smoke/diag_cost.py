"""Diagnostic: measure the actual transaction-cost drag on the momentum
candidate vs the coin-flip null, and confirm the cost cfg is applied.

For AMZN lookback=5:
- full-sample run_bars under zero / realistic / conservative: total return,
  final equity, number of fills, and dollar turnover (sum |delta|*fill_price).
- walk_forward (full series) aggregate mean/median log return under each cfg.
- the same for a coin-flip null signal (random_signals seed 42).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt

WARM = 60
LOOKBACK = 5
ZERO = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
REAL = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM,
                         commission_per_trade=2.0, commission_per_share=0.003,
                         slippage_cents=2.0, slippage_proportional=0.0005)
CONS = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM,
                         commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.0010)


def describe(name, cfg, bars, sig):
    res = bt.run_bars(list(bars), sig, cfg)
    eq = res.equity_curve
    total = (eq[-1] - 1e6) / 1e6
    turnover = sum(abs(f.shares) * f.price for f in res.trades)
    n_fills = len(res.trades)
    wf = bt.walk_forward(list(bars), sig, train_window=252, test_window=84,
                         warmup=WARM, overlap_window=60, cfg=cfg)
    agg = wf.aggregate_metrics
    print("  {:12s} total={:+.4f} eq_final={:.0f} fills={} turnover$={:.0f} "
          "wf_mean={:+.5f} wf_median={:+.5f}".format(
              name, total, eq[-1], n_fills, turnover,
              agg["mean_log_total_return"], agg["median_log_total_return"]))


bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=LOOKBACK)
null = bt.random_signals(len(closes), seed=42)

print("AMZN momentum lookback=5")
for name, cfg in (("zero", ZERO), ("realistic", REAL), ("conservative", CONS)):
    describe(name, cfg, bars, sig)
print("AMZN coin-flip null (seed 42)")
for name, cfg in (("zero", ZERO), ("realistic", REAL), ("conservative", CONS)):
    describe(name, cfg, bars, null)
print("DIAG DONE")