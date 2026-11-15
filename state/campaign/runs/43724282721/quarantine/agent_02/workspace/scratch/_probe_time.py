import time
import sys

sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt

t0 = time.time()
bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)
sig = bt.momentum_signals(closes, lookback=5)
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
    periods_per_year=252,
    min_segment_bars=400,
)
print("one stress_segments call seconds:", round(time.time() - t0, 2))
print("n_folds:", res.n_folds, "verdict:", res.overall_verdict)