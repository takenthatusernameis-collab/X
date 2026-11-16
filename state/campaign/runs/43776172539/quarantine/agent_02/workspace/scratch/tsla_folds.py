import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest.perturbation import random_signals

bars, _ = bt.load_ticker("TSLA")
seg_bars = list(bars)[60:1022]
n = len(seg_bars)
null_sig = random_signals(n, seed=42 + 5 * 1000)

cfgs = {
    "zero": bt.BacktestConfig(initial_capital=1e6),
    "realistic": bt.BacktestConfig(initial_capital=1e6, commission_per_trade=2.0,
                                   commission_per_share=0.003, slippage_cents=2.0,
                                   slippage_proportional=0.0005),
    "conservative": bt.BacktestConfig(initial_capital=1e6, commission_per_trade=4.0,
                                      commission_per_share=0.006, slippage_cents=4.0,
                                      slippage_proportional=0.0010),
}
results = {}
for name, cfg in cfgs.items():
    wf = bt.walk_forward(seg_bars, null_sig, train_window=252, test_window=84,
                         warmup=0, overlap_window=0, cfg=cfg)
    rets = np.array([f.metrics["total_return"] for f in wf.folds])
    results[name] = rets
    print(f"{name:12s} n_folds={len(rets)}")
    print("  total_returns:", [round(float(x), 6) for x in rets])
    logs = np.log1p(np.clip(rets, -1 + 1e-12, None))
    print(f"  median_log={np.median(logs):+.8f} mean_log={logs.mean():+.8f}")

print("\nfold-by-fold deltas (realistic - zero), (conservative - zero):")
for i in range(len(results["zero"])):
    print(f"  fold {i:2d}: zero={results['zero'][i]:+.6f} "
          f"real={results['realistic'][i]:+.6f} (d={results['realistic'][i]-results['zero'][i]:+.6f}) "
          f"cons={results['conservative'][i]:+.6f} (d={results['conservative'][i]-results['zero'][i]:+.6f})")