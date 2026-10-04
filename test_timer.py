import sys
import time
sys.path.insert(0, ".")
import numpy as np
import research.backtest as bt


def ma_crossover_signals(closes: np.ndarray, fast: int, slow: int):
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    fast, slow = int(fast), int(slow)
    for i in range(max(fast, slow) - 1, n):
        fast_ma = np.mean(closes[i - fast + 1 : i + 1])
        slow_ma = np.mean(closes[i - slow + 1 : i + 1])
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if fast_ma > slow_ma else -1.0)
    return signals


bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()
fast, slow = 20, 60
signals = ma_crossover_signals(closes, fast, slow)
warmup = slow
cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=warmup)

start = time.perf_counter()
result = bt.walk_forward(
    list(bars), signals, train_window=252, test_window=84,
    warmup=warmup, overlap_window=60, cfg=cfg,
)
elapsed = time.perf_counter() - start
print(f"walk_forward elapsed: {elapsed:.2f}s, folds={result.aggregate_metrics['n_folds']}, "
      f"oos={result.aggregate_metrics['total_oos_periods']}, "
      f"mean_log={result.aggregate_metrics['mean_log_total_return']:.3f}, "
      f"median_log={result.aggregate_metrics['median_log_total_return']:.3f}")
