import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

np.random.seed(42)
bars = bt.generate_bars(2500, regimes=[
    bt.Regime(drift_annual=0.03, vol_annual=0.18, intraday_range_scale=0.015),
    bt.Regime(drift_annual=-0.01, vol_annual=0.35, intraday_range_scale=0.030),
], p_transition=0.008, start_price=100.0, seed=42)

def volatility_regime_signals(closes, vol_window=20, threshold=0.20):
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    sqrt252 = np.sqrt(252.0)
    for i in range(int(vol_window) - 1, n):
        log_returns = np.log(closes[i - vol_window + 1 : i + 1])
        realized_annual = float(np.std(log_returns, ddof=1)) * sqrt252
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if realized_annual < threshold else 0.0)
    return signals

signals = volatility_regime_signals(bars.closes_array(), 20, 0.20)
nonneutral = sum(1 for s in signals if abs(s.weight) > 1e-12)
print("non-neutral signals:", nonneutral)
print("first non-neutral date:", next(s.date for s in signals if abs(s.weight) > 1e-12))

cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=20)
res = bt.run_bars(list(bars), signals, cfg)
print("trades:", len(res.trades))
print("total_return:", res.equity_curve[-1]/res.equity_curve[0] - 1)
