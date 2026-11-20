import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)
cfg = bt.BacktestConfig(initial_capital=1e6, commission_per_trade=2.0,
                        commission_per_share=0.003, slippage_cents=2.0,
                        slippage_proportional=0.0005)
sig = bt.momentum_signals(closes, lookback=5)
t0 = time.time()
res = bt.stress_segments(signals_fn=bt.momentum_signals, bars=list(bars),
                         signals=sig, param_grid=[{"lookback": 5}],
                         baseline=(("lookback", 5),), segment_fn=seg_fn,
                         train_window=252, test_window=84, warmup=60,
                         overlap_window=60, cfg=cfg, periods_per_year=252,
                         min_segment_bars=400)
dt = time.time() - t0
print("one stress_segments secs", round(dt, 2))
print("verdict", res.overall_verdict, "medians", [round(s.baseline_median_log_return,3) for s in res.scenarios])