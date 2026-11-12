import sys
sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt
import numpy as np

bars, dates = bt.load_ticker("NVDA")
bars_list = list(bars)
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)

c = labels[0]; start = 0; seg = None
for i in range(1, len(labels)+1):
    if i == len(labels) or labels[i] != c:
        if i - start >= 400:
            seg = (c, start, i); break
        c = labels[i]; start = i
name, s, e = seg
seg_bars = bars_list[s:e]
seg_signals = sig[s:e]
seg_closes = closes[s:e]

def cfg_from_costs(costs):
    return bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0,
        commission_per_share=float(costs[0]), commission_per_trade=1.0,
        slippage_cents=float(costs[1]), slippage_proportional=float(costs[2]))

c0 = cfg_from_costs((0.0, 0.0, 0.0))
c1 = cfg_from_costs((0.005, 1.0, 0.001))

print("NVDA segment:", name, s, e, "len", len(seg_bars))

# 1) Direct walk_forward with coin-flip random signals, zero vs base
from research.backtest.perturbation import random_signals
for label, cfg in [("ZERO", c0), ("BASE", c1)]:
    rng = np.random.default_rng(42 + 5*1000)
    coins = [rng.choice([-1.0, 0.0, 1.0], p=[1/3, 1/3, 1/3]) for _ in range(len(seg_bars))]
    coin_sig = [bt.Signal(date=i+1, weight=float(w)) for i, w in enumerate(coins)]
    res = bt.walk_forward(seg_bars, coin_sig, train_window=252, test_window=84,
                          warmup=0, overlap_window=0, cfg=cfg)
    meds = np.log1p(np.clip([f.metrics["total_return"] for f in res.folds], -1+1e-12, None))
    print("  coin flip {}: med={:.4f} medians={}".format(label, float(np.median(meds)),
          [round(float(np.median(meds)),3)]))

# 2) framework noise_benchmark
from research.backtest.perturbation import noise_benchmark
for label, cfg in [("ZERO", c0), ("BASE", c1)]:
    noise = noise_benchmark(
        bars=seg_bars, param_grid=[{"lookback": 5}],
        train_window=252, test_window=84, warmup=0, overlap_window=0,
        periods_per_year=252, cfg=cfg)
    print("  noise_benchmark {} baseline_median={:.4f}".format(label, noise.baseline_median_log_return))
    fold_rets = np.array(noise.noise_fold_median_log_returns, dtype=np.float64)
    print("    first 5 fold medians:", [round(float(x),3) for x in fold_rets[:5]],
          "std:", round(float(np.std(fold_rets)),3))

# 3) full-sample run_bars of the coin-flip null with zero vs base costs
for label, cfg in [("ZERO", c0), ("BASE", c1)]:
    rng = np.random.default_rng(42 + 5*1000)
    coins = [rng.choice([-1.0, 0.0, 1.0], p=[1/3, 1/3, 1/3]) for _ in range(len(bars_list))]
    coin_sig = [bt.Signal(date=i+1, weight=float(w)) for i, w in enumerate(coins)]
    res = bt.run_bars(bars_list, coin_sig, cfg)
    print("  coin flip full-sample {} returns: {:.4f} fills={} commission={:.0f} slippage={:.0f}".format(
        label, (res.equity_curve[-1] - 1e6) / 1e6, len(res.trades),
        sum(f.commission for f in res.trades),
        sum(abs(f.shares) * (cfg.slippage_cents/100.0 + closes[int(f.date)-1]*cfg.slippage_proportional) for f in res.trades)))
