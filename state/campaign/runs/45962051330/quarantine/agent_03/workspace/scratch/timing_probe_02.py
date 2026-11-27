import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

COST = bt.BacktestConfig(
    initial_capital=1e6, warmup_periods=60,
    commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005,
)

bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)
sig = bt.momentum_signals(closes, lookback=5)

t0 = time.time()
res = bt.stress_segments(
    signals_fn=bt.momentum_signals, bars=bars, signals=sig,
    param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
    segment_fn=seg_fn, train_window=252, test_window=84,
    warmup=60, overlap_window=60, periods_per_year=252,
    min_segment_bars=400, cfg=COST,
)
dt = time.time() - t0
print("one stress_segments (realistic cost) seconds:", round(dt, 2))
print("segments:", [s.name for s in res.scenarios])
print("candidate medians:", [round(s.baseline_median_log_return, 4) for s in res.scenarios])
print("null medians:", [round(s.noise_median_log_return, 4) for s in res.scenarios])
print("verdict:", res.overall_verdict, "n_folds:", res.n_folds)