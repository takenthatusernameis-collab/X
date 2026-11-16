import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)
sig = bt.momentum_signals(closes, lookback=5)

zero = bt.BacktestConfig(initial_capital=1e6)
real = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
cons = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=4.0,
    commission_per_share=0.006,
    slippage_cents=4.0,
    slippage_proportional=0.0010,
)

# reconstruct segments like stress_segments
lab = [seg_fn(closes, i) for i in range(len(closes))]
segments = []
cur = lab[0]
start = 0
for i in range(1, len(lab)):
    if lab[i] != cur:
        if i - start >= 400:
            segments.append((cur, start, i))
        cur = lab[i]
        start = i
if len(lab) - start >= 400:
    segments.append((cur, start, len(lab)))
print("segments:", segments)

for name, cfg in [("zero", zero), ("realistic", real), ("conservative", cons)]:
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": 5}],
        baseline=(("lookback", 5),),
        segment_fn=seg_fn,
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=400,
    )
    print(name, "cand medians", [round(s.baseline_median_log_return, 4) for s in res.scenarios],
          "null medians", [round(s.noise_median_log_return, 4) for s in res.scenarios],
          "verdict", res.overall_verdict)

# direct null on first segment
name, s, e = segments[0]
seg_bars = list(bars)[s:e]
for cname, cfg in [("zero", zero), ("realistic", real), ("conservative", cons)]:
    nb = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": 5}],
        train_window=252,
        test_window=84,
        warmup=0,
        overlap_window=0,
        cfg=cfg,
        periods_per_year=252,
    )
    print("null", cname, "median", round(nb.baseline_median_log_return, 4),
          "n_folds", nb.n_folds)

# full-sample cost comparison for the same signal
for cname, cfg in [("zero", zero), ("realistic", real), ("conservative", cons)]:
    r = bt.run_bars(list(bars), sig, cfg)
    tr = (r.equity_curve[-1] - 1e6) / 1e6
    print("full-sample", cname, "total_return", round(tr, 4), "n_fills", len(r.trades))