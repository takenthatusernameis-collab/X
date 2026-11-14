import sys
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

zero = bt.BacktestConfig()
canon = bt.BacktestConfig(
    commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005,
)

for name, cfg in [("zero", zero), ("canon", canon)]:
    res = bt.walk_forward(list(bars), sig, train_window=252, test_window=84,
                          warmup=60, overlap_window=60, cfg=cfg)
    rets = [f.metrics["total_return"] for f in res.folds]
    ntrades = [f.metrics.get("n_trades") for f in res.folds]
    eq0 = res.folds[0].equity_curve
    print(name, "agg median_log_ret", round(res.aggregate_metrics["median_log_total_return"], 5),
          "mean_folds_ret", round(float(np.mean(rets)), 5))
    print("   fold0 total_return", round(rets[0], 5), "n_trades", ntrades[0],
          "eq0_first", round(float(eq0[0]), 2), "eq0_last", round(float(eq0[-1]), 2))
    print("   n_trades per fold:", ntrades)

# turnover: count signal sign flips
flips = int(np.sum(np.diff(np.sign([s.weight for s in sig])) != 0))
print("signal sign flips (full series):", flips, "of", len(sig))