"""Debug: check whether costs are applied to candidate walk-forward medians."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
segments = []
cur = labels[0]
start = 0
for i in range(1, len(labels)):
    if labels[i] != cur:
        segments.append((cur, start, i))
        cur = labels[i]
        start = i
if len(labels) - start >= 400:
    segments.append((cur, start, len(labels)))

seg_name, s, e = segments[1]
seg_bars = list(bars)[s:e]
seg_sig = sig[s:e]
print("segment:", seg_name, "bars:", len(seg_bars))

cfg0 = bt.BacktestConfig()
cfg1 = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

r0 = bt.walk_forward(seg_bars, seg_sig, train_window=252, test_window=84,
                     warmup=60, overlap_window=60, cfg=cfg0)
r1 = bt.walk_forward(seg_bars, seg_sig, train_window=252, test_window=84,
                     warmup=60, overlap_window=60, cfg=cfg1)

m0 = [np.log1p(np.clip(f.metrics["total_return"], -1 + 1e-12, None)) for f in r0.folds]
m1 = [np.log1p(np.clip(f.metrics["total_return"], -1 + 1e-12, None)) for f in r1.folds]
print("zero:     med=%.3f n_folds=%d" % (np.median(m0), len(r0.folds)))
print("realistic: med=%.3f n_folds=%d" % (np.median(m1), len(r1.folds)))
print("d=%.3f" % (np.median(m0) - np.median(m1)))

f0, f1 = r0.folds[0], r1.folds[0]
print("zero fold[0] total_return:", f0.metrics["total_return"], "n_trades:", f0.metrics.get("n_trades"))
print("realistic fold[0] total_return:", f1.metrics["total_return"], "n_trades:", f1.metrics.get("n_trades"))

# Full-sample comparison.
res0 = bt.run_bars(list(bars), sig, cfg0)
res1 = bt.run_bars(list(bars), sig, cfg1)
print("full zero trades:", len(res0.trades))
print("full realistic trades:", len(res1.trades))
if res1.trades:
    print("first realistic fill:", res1.trades[0])
print("full zero equity tail:", [round(float(x), 0) for x in res0.equity_curve[-3:]])
print("full realistic equity tail:", [round(float(x), 0) for x in res1.equity_curve[-3:]])

# Null with costs: does the coin-flip median change?
g0 = noise_benchmark(
    bars=seg_bars, param_grid=[{"lookback": 5}],
    train_window=252, test_window=84, warmup=0, overlap_window=0,
    cfg=cfg0, periods_per_year=252, seed=42)
g1 = noise_benchmark(
    bars=seg_bars, param_grid=[{"lookback": 5}],
    train_window=252, test_window=84, warmup=0, overlap_window=0,
    cfg=cfg1, periods_per_year=252, seed=42)
print("null zero:     baseline=%.3f" % g0.baseline_median_log_return)
print("null realistic: baseline=%.3f" % g1.baseline_median_log_return)
