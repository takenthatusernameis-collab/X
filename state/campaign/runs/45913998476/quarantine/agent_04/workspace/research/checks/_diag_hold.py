import sys
sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt
from research.checks.momentum_hold_period_sensitivity import hold_effective_signals, LOOKBACK

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=LOOKBACK)
print("base signal weights [5:17]:", [s.weight for s in sig[5:17]])

for hold in (1, 2, 3, 5):
    eff = hold_effective_signals(sig, hold)
    weights = [s.weight for s in eff[5:17]]
    cfg = bt.BacktestConfig(initial_capital=1e6)
    res = bt.run_bars(list(bars), eff, cfg)
    tot = (res.equity_curve[-1] - 1e6) / 1e6
    turns = sum(1 for i in range(1, len(eff)) if eff[i].weight != eff[i - 1].weight)
    print(f"hold={hold}: weights[5:17]={weights} total_return={tot:+.6f} turns={turns}")

for hold in (1, 2, 3, 5):
    labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
    seg_fn = bt.segment_fn_from_labels(labels)
    eff = hold_effective_signals(sig, hold)
    res = bt.stress_segments(
        signals_fn=lambda c, h=hold, **p: hold_effective_signals(bt.momentum_signals(c, lookback=LOOKBACK), h),
        bars=list(bars),
        signals=eff,
        param_grid=[{"hold": hold}],
        baseline=(("hold", hold),),
        segment_fn=seg_fn,
        train_window=252, test_window=84, warmup=60, overlap_window=60,
        periods_per_year=252, min_segment_bars=400,
    )
    meds = [round(s.baseline_median_log_return, 6) for s in res.scenarios]
    print(f"WF hold={hold}: medians={meds}")