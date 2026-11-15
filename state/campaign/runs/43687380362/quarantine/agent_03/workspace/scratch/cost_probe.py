import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()

for name, cfg in [
    ("zero", bt.BacktestConfig()),
    ("realistic", bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                                    slippage_cents=2.0, slippage_proportional=0.0005)),
    ("conservative", bt.BacktestConfig(commission_per_trade=4.0, commission_per_share=0.006,
                                       slippage_cents=4.0, slippage_proportional=0.001)),
]:
    sig = bt.momentum_signals(closes, lookback=5)
    res = bt.run_bars(list(bars), sig, cfg)
    n_fills = len(res.trades)
    total_comm = sum(f.commission for f in res.trades)
    print(name, "fills", n_fills, "total_commission", round(total_comm, 2),
          "final_equity", round(float(res.equity_curve[-1]), 2))
    wf = bt.walk_forward(list(bars), sig, train_window=252, test_window=84,
                         warmup=60, overlap_window=60, cfg=cfg)
    lr = [float(np.log1p(np.clip(f.metrics["total_return"], -1 + 1e-12, None))) for f in wf.folds]
    print("   folds", len(lr), "median_log", round(float(np.median(lr)), 5),
          "mean_log", round(float(np.mean(lr)), 5))