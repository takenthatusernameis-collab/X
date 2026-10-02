"""Diagnose fold returns in the MA example walk-forward."""
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
    for i in range(slow - 1, n):
        fm = np.mean(closes[i - fast + 1 : i + 1])
        sm = np.mean(closes[i - slow + 1 : i + 1])
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
    return signals

fast, slow = 20, 60
signals = ma_crossover_signals(bars.closes_array(), fast, slow)
warmup = slow
cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=warmup)

result = bt.walk_forward(list(bars), signals, train_window=252, test_window=84,
                         warmup=warmup, overlap_window=60, cfg=cfg0)
print("n_folds:", result.aggregate_metrics["n_folds"])
for f in result.folds:
    tr = f.metrics["total_return"]
    if tr <= -0.5:
        print("fold", f.fold_index, "start=", f.start_date, "end=", f.end_date,
              "total_return=", repr(tr), "sharpe=", f.metrics["sharpe"],
              "n_trades=", f.metrics["n_trades"],
              "eq0=", f.equity_curve[0], "eq_last=", f.equity_curve[-1],
              "eq_min=", f.equity_curve.min())
