import sys
sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt
import numpy as np

bars, dates = bt.load_ticker("AAPL")
bars_list = list(bars)
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)

c = labels[0]; start = 0; seg = None
for i in range(1, len(labels)+1):
    if i == len(labels) or labels[i] != c:
        if i - start >= 400:
            seg = (c, start, i); break
        c = labels[i]; start = i
name, s, e = seg
seg_bars = bars_list[s:e]
seg_signals = sig[s:e]

def cfg_from_costs(costs):
    return bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0,
        commission_per_share=float(costs[0]), commission_per_trade=1.0,
        slippage_cents=float(costs[1]), slippage_proportional=float(costs[2]))

c0 = cfg_from_costs((0.0, 0.0, 0.0))
c1 = cfg_from_costs((0.005, 1.0, 0.001))
c2 = cfg_from_costs((0.01, 2.0, 0.002))

# Direct walk_forward: compare equity curves fold by fold
print("Per-fold equity (zero vs base vs stressed), last bar of each fold:")
for label, cfg in [("ZERO", c0), ("BASE", c1), ("STRESSED", c2)]:
    res = bt.walk_forward(seg_bars, seg_signals, train_window=252, test_window=84,
                          warmup=60, overlap_window=60, cfg=cfg)
    eqs = np.array([f.equity_curve[-1] for f in res.folds], dtype=np.float64)
    print("  {} first_eq={:.0f} last_eq={:.0f}".format(
        label, eqs[0], eqs[-1]))

# Now: compute total cost paid in base vs zero over ALL folds by comparing
# equity curves bar-by-bar at the OOS end.
print("\nCost impact via equity comparison (last OOS bar of each fold):")
res0 = bt.walk_forward(seg_bars, seg_signals, train_window=252, test_window=84,
                       warmup=60, overlap_window=60, cfg=c0)
res1 = bt.walk_forward(seg_bars, seg_signals, train_window=252, test_window=84,
                       warmup=60, overlap_window=60, cfg=c1)
eq0 = np.array([f.equity_curve[-1] for f in res0.folds], dtype=np.float64)
eq1 = np.array([f.equity_curve[-1] for f in res1.folds], dtype=np.float64)
diff = eq1 - eq0
print("  sum of equity differences (base - zero) over {} folds: {:.2f}".format(len(eq0), diff.sum()))
print("  mean diff per fold: {:.2f}  (== {:.4f} of $1M)".format(diff.mean(), diff.mean()/1e6))
log_diff = np.log1p(np.clip(eq1/1e6, -1+1e-12, None)) - np.log1p(np.clip(eq0/1e6, -1+1e-12, None))
print("  mean log-return drag per fold: {:.4f}".format(log_diff.mean()))

# Also compute what the cost SHOULD be: run one fold manually and count fills.
from research.backtest.engine import Backtest, run_bars
full_res0 = run_bars(seg_bars, seg_signals, c0)
full_res1 = run_bars(seg_bars, seg_signals, c1)
print("\nFull-segment run (336+ bars):")
print("  zero  fills={} commission={:.2f} slippage={:.2f}".format(
    len(full_res0.trades), sum(f.commission for f in full_res0.trades),
    sum(f.shares * c0.slippage_cents/100.0 + abs(f.shares)*closes[s+f.date-1]*c0.slippage_proportional
        for f in full_res0.trades)))
print("  base  fills={} commission={:.2f} slippage={:.2f}".format(
    len(full_res1.trades), sum(f.commission for f in full_res1.trades),
    sum(f.shares * c1.slippage_cents/100.0 + abs(f.shares)*closes[s+f.date-1]*c1.slippage_proportional
        for f in full_res1.trades)))
print("  base-vs-zero extra commission: {:.2f}  extra slippage: {:.2f}".format(
    sum(f.commission for f in full_res1.trades) - sum(f.commission for f in full_res0.trades),
    sum(f.shares * c1.slippage_cents/100.0 + abs(f.shares)*closes[s+f.date-1]*c1.slippage_proportional
        for f in full_res1.trades) - sum(f.shares * c0.slippage_cents/100.0 + abs(f.shares)*closes[s+f.date-1]*c0.slippage_proportional
        for f in full_res0.trades)))
