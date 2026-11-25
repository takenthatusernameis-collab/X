import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
import numpy as np

bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

def cfg(m):
    return bt.BacktestConfig(
        initial_capital=1e6, warmup_periods=0,
        commission_per_trade=2.0*m, commission_per_share=0.003*m,
        slippage_cents=2.0*m, slippage_proportional=0.0005*m)

for m in (0.0, 1.0, 5.0):
    res = bt.walk_forward(list(bars), sig, train_window=252, test_window=84,
                          warmup=60, overlap_window=60, cfg=cfg(m))
    folds = res.folds
    tots = [f.metrics["total_return"] for f in folds[:5]]
    # count trades in first fold OOS
    print(f"m={m}: n_folds={len(folds)} mean_log={res.aggregate_metrics['mean_log_total_return']:+.4f} "
          f"median_log={res.aggregate_metrics['median_log_total_return']:+.4f}")
    print(f"   first5 total_return={['%.4f'%t for t in tots]}")

# Full-sample single run to see trade count and cost drag
for m in (0.0, 5.0):
    r = bt.run_bars(list(bars), sig, cfg(m))
    n_trades = len(r.trades)
    total_comm = sum(t.commission for t in r.trades)
    eq0 = r.equity_curve[0]; eqT = r.equity_curve[-1]
    print(f"full-sample m={m}: n_fills={n_trades} total_commission={total_comm:.1f} "
          f"total_return={(eqT-eq0)/eq0:+.4f}")