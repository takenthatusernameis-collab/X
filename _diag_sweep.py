"""Diagnose fold returns across the perturbation grid."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

np.random.seed(42)
bars = bt.generate_bars(2500, regimes=[
    bt.Regime(drift_annual=0.03, vol_annual=0.18, intraday_range_scale=0.015),
    bt.Regime(drift_annual=-0.01, vol_annual=0.35, intraday_range_scale=0.030),
], p_transition=0.008, start_price=100.0)

def ma_crossover_signals(closes, fast, slow):
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(int(slow) - 1, n):
        fm = np.mean(closes[i - int(fast) + 1 : i + 1])
        sm = np.mean(closes[i - int(slow) + 1 : i + 1])
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
    return signals

fast, slow = 20, 60
grid = bt.parameter_grid_around((("fast", fast), ("slow", slow)),
                                multipliers=(0.5, 1.0, 2.0))
cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=slow)

sweep = bt.parameter_sweep(
    lambda closes, **p: ma_crossover_signals(closes, p["fast"], p["slow"]),
    bars=list(bars), param_grid=grid, train_window=252, test_window=84,
    warmup=slow, overlap_window=60, cfg=cfg0)

for ps, returns in zip(sweep.param_sets, sweep.fold_total_returns):
    bad = [r for r in returns if r <= -1.0]
    print(ps, "bad_returns_le_-1:", bad, "log1p_issues:",
          sum(np.isinf(np.log1p(np.array(returns))) + np.isnan(np.log1p(np.array(returns)))))
