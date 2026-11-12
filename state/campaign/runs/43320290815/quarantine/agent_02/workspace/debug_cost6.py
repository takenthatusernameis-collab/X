import sys
sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt
import numpy as np

bars, dates = bt.load_ticker("NVDA")
bars_list = list(bars)
closes = bars.closes_array()

def cfg_from_costs(costs):
    return bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0,
        commission_per_share=float(costs[0]), commission_per_trade=1.0,
        slippage_cents=float(costs[1]), slippage_proportional=float(costs[2]))

c0 = cfg_from_costs((0.0, 0.0, 0.0))
c1 = cfg_from_costs((0.005, 1.0, 0.001))

rng = np.random.default_rng(42 + 5*1000)
coins = [rng.choice([-1.0, 0.0, 1.0], p=[1/3, 1/3, 1/3]) for _ in range(len(bars_list))]
coin_sig = [bt.Signal(date=i+1, weight=float(w)) for i, w in enumerate(coins)]

seg_bars = bars_list[60:1116]
seg_signals = [bt.Signal(date=i+1, weight=float(w)) for i, w in enumerate(coins[60:1116])]
seg_closes = closes[60:1116]

print("=== run_bars full segment ===")
for label, cfg in [("ZERO", c0), ("BASE", c1)]:
    res = bt.run_bars(seg_bars, seg_signals, cfg)
    total_commission = sum(f.commission for f in res.trades)
    total_slippage = sum(abs(f.shares) * (cfg.slippage_cents/100.0 + closes[int(f.date)-1]*cfg.slippage_proportional)
                         for f in res.trades)
    fills_list = list(res.trades)
    print("  {}: fills={} total_return={:.3f} eq_end={:.0f} commission={:.2f} slippage={:.2f}".format(
        label, len(fills_list), (res.equity_curve[-1]-1e6)/1e6, res.equity_curve[-1],
        total_commission, total_slippage))
    if label == "BASE":
        print("    first 3 fills (date, shares, price, commission):")
        for f in fills_list[:3]:
            print("      date={} shares={:.1f} price={:.2f} commission={:.2f}".format(
                f.date, f.shares, f.price, f.commission))
        print("    first 3 bars closes:", [round(float(b.close),2) for b in seg_bars[:3]])

print("\n=== walk_forward segment (warmup=0, train=252, test=84, overlap=0) ===")
for label, cfg in [("ZERO", c0), ("BASE", c1)]:
    res = bt.walk_forward(seg_bars, seg_signals, train_window=252, test_window=84,
                          warmup=0, overlap_window=0, cfg=cfg)
    meds = []
    for f in res.folds:
        tr = sum(len(t.trades) for t in [f])
        meds.append(float(np.log1p(np.clip(f.metrics["total_return"], -1+1e-12, None))))
    print("  {}: n_folds={} fold_medians={}".format(label, len(res.folds), [round(x,3) for x in meds]))
    # equity at OOS end of each fold
    eq = [f.equity_curve[-1] for f in res.folds]
    print("     eq_end_folds=[{:.0f}, {:.0f}, {:.0f}, {:.0f}, {:.0f}]".format(*eq[:5], *(eq[-1:])))