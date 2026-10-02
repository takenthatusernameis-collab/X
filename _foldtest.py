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
fold_start = 0
fold_bars = list(bars)[fold_start : fold_start + 20 + 252 + 84]
fold_signals = signals[fold_start : fold_start + 20 + 252 + 84]
print("fold bar dates:", fold_bars[0].date, "to", fold_bars[-1].date)
print("signals dates:", fold_signals[0].date, "to", fold_signals[-1].date)
print("first 25 signal weights:", [round(s.weight, 2) for s in fold_signals[:25]])
fold_cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=20)
res = bt.run_bars(list(fold_bars), fold_signals, fold_cfg)
print("trades:", len(res.trades))
print("positions:", [round(p,1) for p in res.positions[15:26]])
print("equity[15:26]:", [round(e,1) for e in res.equity_curve[15:26]])
print("first trades:", res.trades[:5])
