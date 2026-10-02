"""Check determinism of the backtest across repeated runs."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt

def run_once():
    np.random.seed(42)
    bars = bt.generate_bars(2500, regimes=[
        bt.Regime(drift_annual=0.03, vol_annual=0.18, intraday_range_scale=0.015),
        bt.Regime(drift_annual=-0.01, vol_annual=0.35, intraday_range_scale=0.030),
    ], p_transition=0.008, start_price=100.0)

    def ma_signals(closes, fast, slow):
        n = len(closes)
        signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
        for i in range(int(slow) - 1, n):
            fm = np.mean(closes[i - int(fast) + 1 : i + 1])
            sm = np.mean(closes[i - int(slow) + 1 : i + 1])
            signals[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
        return signals

    signals = ma_signals(bars.closes_array(), 20, 60)
    res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6, warmup_periods=60))
    return res.equity_curve[-1], len(res.trades), [f.date for f in res.trades[:3]]

for r in range(5):
    eq, ntr, first3 = run_once()
    print("run", r, "final_equity=", eq, "trades=", ntr, "first_trades=", first3)
