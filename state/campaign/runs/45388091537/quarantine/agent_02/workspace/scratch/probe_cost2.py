import sys
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt

def cfg_real():
    return bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                             slippage_cents=2.0, slippage_proportional=0.0005)

def cfg_extreme():
    return bt.BacktestConfig(commission_per_trade=200.0, commission_per_share=0.3,
                             slippage_cents=200.0, slippage_proportional=0.05)

def run_full(ticker, lookback, cfg):
    bars = bt.load_ticker(ticker)[0]
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=lookback)
    res = bt.run_bars(list(bars), sig, cfg)
    tr = res.equity_curve[-1] / res.equity_curve[0] - 1.0
    # turnover
    fills = res.trades
    notional = sum(abs(f.shares) * f.price for f in fills)
    avg_eq = float(np.mean(res.equity_curve))
    turn = notional / avg_eq if avg_eq > 0 else 0.0
    n_trades = len(fills)
    # signal flips
    weights = [s.weight for s in sig]
    flips = sum(1 for i in range(1, len(weights)) if weights[i] != weights[i-1] and weights[i] != 0.0 and weights[i-1] != 0.0)
    nonzero = sum(1 for w in weights if w != 0.0)
    return tr, turn, n_trades, flips, nonzero, bars.n_bars

for ticker in ["AAPL", "TSLA", "JPM"]:
    for lb in [3, 5, 10]:
        tr0, turn0, nt0, fl, nz, n = run_full(ticker, lb, bt.BacktestConfig())
        tr1, turn1, nt1, _, _, _ = run_full(ticker, lb, cfg_real())
        tr2, turn2, nt2, _, _, _ = run_full(ticker, lb, cfg_extreme())
        print(f"{ticker} lb={lb}: bars={n} nonzero={nz} flips={fl}")
        print(f"   zero:      total_ret={tr0:+.4f} turnover={turn0:.2f} trades={nt0}")
        print(f"   realistic: total_ret={tr1:+.4f} turnover={turn1:.2f} trades={nt1}")
        print(f"   extreme:   total_ret={tr2:+.4f} turnover={turn2:.2f} trades={nt2}")