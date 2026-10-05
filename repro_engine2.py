import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import research.backtest as bt

# tiny deterministic family, 8 bars
rng = np.random.default_rng(0)
fam = []
prices = {a: np.array([100.0 + rng.random() * 10 for _ in range(8)], dtype=np.float64) for a in range(3)}
for a in range(3):
    dates = np.arange(1, 9, dtype=np.int64)
    c = prices[a]
    h = np.maximum(c, np.roll(c, -1)); l = np.minimum(c, np.roll(c, -1))
    fam.append(bt.BarSequence(dates, c, h, l, c, np.full(8, 1e6)))

pnl = bt.csrs_spread_family(fam, 2, 1, 1)
synth = bt.build_synthetic_spread_asset(pnl, start_price=1e6)
close = synth.closes_array()

# direct: comp over all bars from 1 to 7
comp_direct = 1.0
for k in range(1, len(pnl)):
    comp_direct *= (1.0 + pnl[k])
print("direct compound:", comp_direct)

# engine with constant-shares weight: total_return over full series
signals = [bt.Signal(date=i + 1, weight=close[i] / close[0]) for i in range(len(pnl))]
wf = bt.walk_forward(list(synth), signals, train_window=3, test_window=3, warmup=0, overlap_window=0,
                     cfg=bt.BacktestConfig())
print("engine folds:", len(wf.folds))
if wf.folds:
    print("engine total_return:", wf.folds[0].metrics["total_return"])
    print("match (abs tol 1e-9):", np.isclose(wf.folds[0].metrics["total_return"], comp_direct - 1, atol=1e-9))

# engine self-consistency with constant-shares: equity[i] == equity[i-1]*(1+pnl[i])
res = bt.run_bars(list(synth), signals, bt.BacktestConfig(initial_capital=1e6))
consistent = all(np.isclose(res.equity_curve[i], res.equity_curve[i - 1] * (1.0 + pnl[i]))
                 for i in range(1, len(pnl)))
print("engine self-consistency (compound):", consistent)
