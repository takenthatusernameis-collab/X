import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark, random_signals

bars, dates = bt.load_ticker("TSLA")
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)
sig = bt.momentum_signals(closes, lookback=5)

zero = bt.BacktestConfig(initial_capital=1e6)
real = bt.BacktestConfig(initial_capital=1e6, commission_per_trade=2.0,
                         commission_per_share=0.003, slippage_cents=2.0,
                         slippage_proportional=0.0005)
cons = bt.BacktestConfig(initial_capital=1e6, commission_per_trade=4.0,
                         commission_per_share=0.006, slippage_cents=4.0,
                         slippage_proportional=0.0010)

# reconstruct segments
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
print("TSLA segments:", segments)

for name, cfg in [("zero", zero), ("realistic", real), ("conservative", cons)]:
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals, bars=bars, signals=sig,
        param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
        segment_fn=seg_fn, train_window=252, test_window=84, warmup=60,
        overlap_window=60, cfg=cfg, periods_per_year=252, min_segment_bars=400)
    print(name, "cand", [round(s.baseline_median_log_return, 6) for s in res.scenarios],
          "null", [round(s.noise_median_log_return, 6) for s in res.scenarios],
          "n_folds", [s.n_folds for s in res.scenarios])

# direct null per segment
for lab_name, s, e in segments:
    seg_bars = list(bars)[s:e]
    print(f"\nsegment {lab_name} [{s},{e}) n={e-s}")
    # show that the null signal is identical across cfg
    ns = random_signals(e - s, seed=42 + 5 * 1000)
    w = [x.weight for x in ns]
    print("  null weights first 20:", w[:20], "sum abs:", sum(abs(x) for x in w))
    for cname, cfg in [("zero", zero), ("realistic", real), ("conservative", cons)]:
        nb = noise_benchmark(bars=seg_bars, param_grid=[{"lookback": 5}],
                             train_window=252, test_window=84, warmup=0,
                             overlap_window=0, cfg=cfg, periods_per_year=252)
        print(f"  null {cname:12s} median={nb.baseline_median_log_return:+.8f} n_folds={nb.n_folds}")