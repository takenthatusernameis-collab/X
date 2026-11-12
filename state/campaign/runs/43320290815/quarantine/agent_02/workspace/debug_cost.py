import sys
sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt
import numpy as np

bars, dates = bt.load_ticker("AAPL")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)
print("bars:", bars.n_bars, "signals:", len(sig))

def run(cfg, label):
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals, bars=bars, signals=sig,
        param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
        segment_fn=bt.segment_fn_from_labels(
            bt.volatility_blocks(closes, n_blocks=4, window=60)),
        train_window=252, test_window=84, warmup=60, overlap_window=60,
        cfg=cfg, periods_per_year=252, min_segment_bars=400)
    scen = res.scenarios[0]
    segs = [(round(s.baseline_median_log_return,3), round(s.noise_median_log_return,3)) for s in res.scenarios]
    print("{}: segs={} verdict={} n_folds={}".format(
        label, segs, res.overall_verdict, res.n_folds))
    return res

c0 = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0, commission_per_share=0.0,
                       commission_per_trade=1.0, slippage_cents=0.0, slippage_proportional=0.0)
c1 = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0, commission_per_share=0.005,
                       commission_per_trade=1.0, slippage_cents=1.0, slippage_proportional=0.001)
c2 = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0, commission_per_share=0.01,
                       commission_per_trade=1.0, slippage_cents=2.0, slippage_proportional=0.002)

run(c0, "ZERO")
run(c1, "BASE")
run(c2, "STRESSED")

res0 = bt.run_bars(list(bars), sig, c0)
res1 = bt.run_bars(list(bars), sig, c1)
print("\nfull-sample fills: zero={:.0f} trips, base={:.0f} trips".format(len(res0.trades), len(res1.trades)))
print("zero  total commission: {:.2f}  equity[-1]: {:.2f}".format(sum(f.commission for f in res0.trades), res0.equity_curve[-1]))
print("base  total commission: {:.2f}  equity[-1]: {:.2f}".format(sum(f.commission for f in res1.trades), res1.equity_curve[-1]))
print("base sample fills (first 3):")
for f in res1.trades[:3]:
    print("   date={} shares={:.0f} price={:.2f} commission={:.3f}".format(
        f.date, f.shares, f.price, f.commission))
