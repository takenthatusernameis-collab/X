import sys, time
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt

print("numpy", np.__version__)
t0 = time.time()
bars = bt.load_ticker("AAPL")[0]
print("AAPL bars:", bars.n_bars, "load %.2fs" % (time.time() - t0))

closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)
sig = bt.momentum_signals(closes, lookback=5)
print("signals:", len(sig))

cfg = bt.BacktestConfig(
    commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005,
)
t0 = time.time()
res = bt.stress_segments(
    signals_fn=bt.momentum_signals, bars=bars, signals=sig,
    param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
    segment_fn=seg_fn, train_window=252, test_window=84, warmup=60,
    overlap_window=60, periods_per_year=252, min_segment_bars=400, cfg=cfg,
)
print("stress_segments realistic: %.2fs verdict=%s" % (time.time() - t0, res.overall_verdict))
print("  medians:", [round(s.baseline_median_log_return, 4) for s in res.scenarios])
print("  nulls:  ", [round(s.noise_median_log_return, 4) for s in res.scenarios])
print("  n_folds:", res.n_folds)

t0 = time.time()
res0 = bt.stress_segments(
    signals_fn=bt.momentum_signals, bars=bars, signals=sig,
    param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
    segment_fn=seg_fn, train_window=252, test_window=84, warmup=60,
    overlap_window=60, periods_per_year=252, min_segment_bars=400,
)
print("stress_segments zero-cost: %.2fs verdict=%s" % (time.time() - t0, res0.overall_verdict))
print("  medians:", [round(s.baseline_median_log_return, 4) for s in res0.scenarios])