import sys, time
sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)
sig = bt.momentum_signals(closes, lookback=5)
print("bars", bars.n_bars, "nonneutral", sum(1 for s in sig if abs(s.weight) > 1e-12))

for name, cfg in [
    ("zero", bt.BacktestConfig()),
    ("realistic", bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                                    slippage_cents=2.0, slippage_proportional=0.0005)),
]:
    t0 = time.time()
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals, bars=bars, signals=sig,
        param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
        segment_fn=seg_fn, train_window=252, test_window=84, warmup=60,
        overlap_window=60, periods_per_year=252, min_segment_bars=400, cfg=cfg)
    dt = time.time() - t0
    print(name, "segments", [s.name for s in res.scenarios],
          "medians", [round(s.baseline_median_log_return, 3) for s in res.scenarios],
          "null", [round(s.noise_median_log_return, 3) for s in res.scenarios],
          "verdict", res.overall_verdict, "secs", round(dt, 2))