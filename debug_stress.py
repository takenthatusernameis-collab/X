import sys
import time
sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt
from research.data.preflight import load_manifest

manifest = load_manifest()
bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()

labels = ["segA"] * (len(closes) // 2) + ["segB"] * (len(closes) - len(closes) // 2)
s, e = 0, len(closes) // 2
bars_list = list(bars)
seg_bars = bars_list[s:e]
seg_signals = [bt.Signal(date=i + 1, weight=(1.0 if i % 2 == 0 else 0.0)) for i in range(len(seg_bars))]

grid = [{"fast": 20, "slow": 60}]
cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=60)

print("candidate sweep start")
t0 = time.time()
sweep = bt.parameter_sweep(
    signals_fn=lambda closes, **p: [bt.Signal(date=i + 1, weight=1.0 if i % 3 == 0 else 0.0) for i in range(len(closes))],
    bars=seg_bars,
    param_grid=grid,
    train_window=252,
    test_window=84,
    warmup=60,
    overlap_window=60,
    cfg=cfg0,
)
print("candidate sweep elapsed:", round(time.time() - t0, 2), "s, folds:", sweep.n_folds)

print("noise benchmark start")
t0 = time.time()
noise = bt.noise_benchmark(
    seg_bars,
    param_grid=grid,
    train_window=252,
    test_window=84,
    warmup=0,
    overlap_window=0,
    cfg=cfg0,
)
print("noise benchmark elapsed:", round(time.time() - t0, 2), "s")
