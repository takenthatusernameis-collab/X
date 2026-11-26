import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
runs = []
cur = labels[0]; start = 0
for i in range(1, len(labels)):
    if labels[i] != cur:
        if i - start >= 400:
            runs.append((cur, start, i))
        cur = labels[i]; start = i
if len(labels) - start >= 400:
    runs.append((cur, start, len(labels)))
lab, s, e = runs[0]
seg_bars = list(bars)[s:e]
seg_sig = sig[s:e]

cfgs = {
    "zero": bt.BacktestConfig(),
    "real": bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005),
}
res = {}
for name, cfg in cfgs.items():
    res[name] = bt.walk_forward(seg_bars, seg_sig, train_window=252, test_window=84, warmup=60, overlap_window=60, cfg=cfg)
    print(name, "agg median", round(res[name].aggregate_metrics["median_log_total_return"], 5),
          "mean", round(res[name].aggregate_metrics["mean_log_total_return"], 5))
for i in range(6):
    z = res["zero"].folds[i].metrics["total_return"]
    r = res["real"].folds[i].metrics["total_return"]
    print("fold", i, "zero", round(z, 5), "real", round(r, 5), "diff", round(r - z, 6),
          "ntrades z/r", res["zero"].folds[i].metrics["n_trades"], res["real"].folds[i].metrics["n_trades"])

# Now test a fold where n_trades is high, e.g. fold 0 (6 trades) - check equity diff
print("fold0 equity zero end", round(res["zero"].folds[0].equity_curve[-1], 2),
      "real end", round(res["real"].folds[0].equity_curve[-1], 2))