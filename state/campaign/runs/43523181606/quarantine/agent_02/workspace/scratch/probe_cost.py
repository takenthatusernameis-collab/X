import sys, time
from pathlib import Path
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)

for bps in (0.0, 10.0, 30.0):
    cfg = bt.BacktestConfig(initial_capital=1e6, slippage_proportional=bps / 10000.0)
    t0 = time.time()
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=list(bars),
        signals=bt.momentum_signals(closes, lookback=5),
        param_grid=[{"lookback": 5}],
        baseline=(("lookback", 5),),
        segment_fn=seg_fn,
        train_window=252, test_window=84, warmup=60, overlap_window=60,
        cfg=cfg, periods_per_year=252, min_segment_bars=400,
    )
    dt = time.time() - t0
    print("bps", bps, "segments", [s.name for s in res.scenarios],
          "medians", [round(s.baseline_median_log_return, 4) for s in res.scenarios],
          "nulls", [round(s.noise_median_log_return, 4) for s in res.scenarios],
          "verdict", res.overall_verdict, "secs", round(dt, 2))

sig = bt.momentum_signals(closes, lookback=5)
for bps in (0.0, 10.0, 30.0):
    cfg = bt.BacktestConfig(initial_capital=1e6, slippage_proportional=bps / 10000.0)
    r = bt.run_bars(list(bars), sig, cfg)
    fills = r.trades
    gross = sum(abs(f.shares) * f.price for f in fills) / 1e6
    print("full-sample bps", bps, "n_fills", len(fills),
          "turnover_per_bar_pct", round(100 * gross / len(bars), 4),
          "commissions", round(sum(f.commission for f in fills), 2))