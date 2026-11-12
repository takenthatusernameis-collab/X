import sys
sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt
import numpy as np

bars, dates = bt.load_ticker("AAPL")
bars_list = list(bars)
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)

# Manually replicate what stress_segments does for the first segment.
# The segment must have >= 400 bars; find the first qualifying one.
c = labels[0]; start = 0
seg = None
for i in range(1, len(labels)+1):
    if i == len(labels) or labels[i] != c:
        if i - start >= 400:
            seg = (c, start, i)
            break
        c = labels[i]; start = i
print("first segment:", seg)
if seg is None:
    print("no segment >= 400 bars in first block; using first 400 bars of series")
    s, e = 0, 400
    name = "test"
else:
    name, s, e = seg
seg_bars = bars_list[s:e]
seg_signals = sig[s:e]
closes_seg = seg_bars[-1].__class__ if False else closes[s:e]

def _param_seed(params, base_seed=42):
    total = sum(float(v) for v in params.values())
    return base_seed + int(round(total)) * 1000

def random_signals(n, seed=None):
    rng = np.random.default_rng(seed)
    weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1/3, 1/3, 1/3])
    return [bt.Signal(date=i+1, weight=float(w)) for i, w in enumerate(weights)]

def cfg_from_costs(costs):
    return bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0,
        commission_per_share=float(costs[0]), commission_per_trade=1.0,
        slippage_cents=float(costs[1]), slippage_proportional=float(costs[2]))

c0 = cfg_from_costs((0.0, 0.0, 0.0))
c1 = cfg_from_costs((0.005, 1.0, 0.001))

from research.backtest.perturbation import parameter_sweep
from research.backtest.engine import walk_forward

print("\n--- Direct walk_forward on segment ---")
for label, cfg in [("ZERO", c0), ("BASE", c1)]:
    res = walk_forward(seg_bars, seg_signals, train_window=252, test_window=84,
                       warmup=60, overlap_window=60, cfg=cfg)
    m = np.log1p(np.clip([f.metrics["total_return"] for f in res.folds], -1+1e-12, None))
    print("  {} med={} n_folds={}".format(
        label, round(float(np.median(m)), 4), len(res.folds)))

print("\n--- parameter_sweep (as used inside stress_segments) ---")
for label, cfg in [("ZERO", c0), ("BASE", c1)]:
    sweep = parameter_sweep(
        signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=5),
        bars=seg_bars, param_grid=[{"lookback": 5}],
        train_window=252, test_window=84, warmup=60, overlap_window=60, cfg=cfg)
    m = np.log1p(np.clip(np.array(sweep.fold_total_returns[0], dtype=np.float64), -1+1e-12, None))
    print("  {} med={} fold_returns[0:3]={}".format(label, round(float(np.median(m)), 4),
          [round(x,3) for x in sweep.fold_total_returns[0][:3]]))

print("\n--- noise_benchmark (as used inside stress_segments) ---")
for label, cfg in [("ZERO", c0), ("BASE", c1)]:
    noise = bt.noise_benchmark(
        bars=seg_bars, param_grid=[{"lookback": 5}],
        train_window=252, test_window=84, warmup=0, overlap_window=0, cfg=cfg)
    print("  {} noise_median={}".format(label, noise.baseline_median_log_return))
