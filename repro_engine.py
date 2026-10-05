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
print("pnl:", [round(x, 6) for x in pnl])

synth = bt.build_synthetic_spread_asset(pnl, start_price=1e6)
print("synth closes[0..3]:", [round(synth.closes_array()[i], 4) for i in range(4)])

signals = [bt.Signal(date=i + 1, weight=1.0) for i in range(len(pnl))]
res = bt.run_bars(list(synth), signals, bt.BacktestConfig(initial_capital=1e6))
print("equity_curve[0..4]:", [round(x, 4) for x in res.equity_curve[:5]])

for i in range(1, min(6, len(pnl))):
    lhs = res.equity_curve[i]
    rhs = res.equity_curve[i - 1] * (1.0 + pnl[i])
    print(f"i={i}: engine_equity={lhs:.6f} vs equity_prev*(1+pnl)={rhs:.6f} diff={lhs-rhs:+.2e}")
