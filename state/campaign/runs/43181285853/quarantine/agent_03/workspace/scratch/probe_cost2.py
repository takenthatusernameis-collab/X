import sys
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)
sig = bt.momentum_signals(closes, lookback=5)

cfgs = {
    "zero": bt.BacktestConfig(),
    "realistic": bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                                   slippage_cents=2.0, slippage_proportional=0.0005),
}
for name, cfg in cfgs.items():
    res = bt.run_bars(list(bars), sig, cfg)
    total = (res.equity_curve[-1] - 1e6) / 1e6
    print(name, "fills", len(res.trades), "total_return", round(total, 4))

for name, cfg in cfgs.items():
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals, bars=bars, signals=sig,
        param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
        segment_fn=seg_fn, train_window=252, test_window=84, warmup=60,
        overlap_window=60, periods_per_year=252, min_segment_bars=400, cfg=cfg)
    print(name)
    for s in res.scenarios:
        print("   ", s.name, "cand", repr(s.baseline_median_log_return),
              "null", repr(s.noise_median_log_return))