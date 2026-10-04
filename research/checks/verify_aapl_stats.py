"""Independent verification of the AAPL walk-forward statistics.

Re-runs the MA crossover on real AAPL data and computes fold-return
statistics (mean/median/std, positive-fold count, and a t-statistic against
H0: mean log fold return = 0). Output is printed for inclusion in the
activation record.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt


def ma_crossover_signals(closes, fast, slow):
    n = len(closes)
    s = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(max(fast, slow) - 1, n):
        fm = np.mean(closes[i - fast + 1 : i + 1])
        sm = np.mean(closes[i - slow + 1 : i + 1])
        s[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
    return s


def main():
    np.random.seed(42)
    bars, dates = bt.load_ticker("AAPL")
    cfg = bt.BacktestConfig(warmup_periods=60)
    res = bt.walk_forward(
        list(bars), ma_crossover_signals(bars.closes_array(), 20, 60),
        train_window=252, test_window=84, warmup=60, overlap_window=60, cfg=cfg,
    )
    rets = np.array([float(f.metrics["total_return"]) for f in res.folds], dtype=np.float64)
    log = np.log1p(np.clip(rets, -1.0 + 1e-12, None))
    n = len(log)
    se = 1.96 * log.std(ddof=1) / np.sqrt(n)
    tstat = log.mean() / (log.std(ddof=1) / np.sqrt(n))
    print(f"folds={n}, mean_log_ret={log.mean():.4f}, median_log_ret={np.median(log):.4f}, "
          f"std_log_ret={log.std(ddof=1):.4f}")
    print(f"positive_folds={(log > 0).sum()}/{n}")
    print(f"95_percent_CI_mean={log.mean() - se:.4f}..{log.mean() + se:.4f}")
    print(f"t_statistic_H0_mean_log_zero={tstat:.2f}, df={n - 1}")


if __name__ == "__main__":
    main()
