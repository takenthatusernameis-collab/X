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
result = bt.walk_forward(list(bars), signals, train_window=252, test_window=84,
                         warmup=20, overlap_window=60,
                         cfg=bt.BacktestConfig(initial_capital=1e6, warmup_periods=20))
print("n_folds:", result.aggregate_metrics["n_folds"])
for fold in result.folds[:5]:
    m = fold.metrics
    print(f"fold {fold.fold_index} oos={fold.start_date}..{fold.end_date} "
          f"trades={m['n_trades']} ret={m['total_return']:.4f} "
          f"pos={m['final_position_shares']:.1f} trades_list={[(f.date, f.shares) for f in fold.metrics.get('_trades', [])]}")
