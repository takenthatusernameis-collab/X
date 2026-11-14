import sys
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

zero = bt.BacktestConfig()
canon = bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                          slippage_cents=2.0, slippage_proportional=0.0005)

for name, cfg in [("zero", zero), ("canon", canon)]:
    res = bt.run_bars(list(bars), sig, cfg)
    fills = res.trades
    n = len(fills)
    gross_traded = float(np.sum(np.abs([f.shares for f in fills])))
    total_comm = float(np.sum([f.commission for f in fills]))
    # dollar notional traded (approx): sum |shares|*price
    dollar = float(np.sum([abs(f.shares) * f.price for f in fills]))
    print(f"{name}: n_fills={n} gross_abs_shares={gross_traded:.1f} "
          f"dollar_notional_traded={dollar:.0f} total_commission={total_comm:.2f}")
    print(f"   first 5 fill shares: {[round(f.shares,2) for f in fills[:5]]}")
    print(f"   equity final: {res.equity_curve[-1]:.2f}")

# signal flips over full series
w = np.array([s.weight for s in sig])
print("weight value counts:", {v: int(np.sum(w == v)) for v in (-1.0, 0.0, 1.0)})
print("sign flips:", int(np.sum(np.diff(np.sign(w)) != 0)))

# a genuinely fluctuating signal for contrast: correct return-based momentum
def correct_mom(closes, lookback=5):
    n = len(closes)
    out = [bt.Signal(date=i+1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        r = float(np.log(closes[i] / closes[i - lookback]))
        out[i] = bt.Signal(date=i+1, weight=1.0 if r > 0 else -1.0)
    return out

csig = correct_mom(closes, 5)
cw = np.array([s.weight for s in csig])
print("CORRECT momentum weight counts:", {v: int(np.sum(cw == v)) for v in (-1.0, 0.0, 1.0)})
print("CORRECT sign flips:", int(np.sum(np.diff(np.sign(cw)) != 0)))
cres0 = bt.run_bars(list(bars), csig, zero)
cres1 = bt.run_bars(list(bars), csig, canon)
print("CORRECT full-series equity zero:", round(float(cres0.equity_curve[-1]), 2),
      "canon:", round(float(cres1.equity_curve[-1]), 2))
print("CORRECT n_fills:", len(cres1.trades))