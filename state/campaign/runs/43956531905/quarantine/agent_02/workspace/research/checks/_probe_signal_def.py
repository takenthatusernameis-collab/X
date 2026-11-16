"""Diagnostic: inspect momentum signal definition and turnover (scratch helper)."""
import sys
import numpy as np
sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt

for ticker in ["AAPL", "AMZN", "JPM", "XOM"]:
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    for lb in (3, 5, 10):
        sig = bt.momentum_signals(closes, lookback=lb)
        w = np.array([s.weight for s in sig])
        # after the lookback warmup window
        wa = w[lb:]
        npos = int(np.sum(wa > 0)); nneg = int(np.sum(wa < 0)); nzero = int(np.sum(wa == 0))
        flips = int(np.sum(np.abs(np.diff(wa)) > 0))
        # mean of log price vs actual lookback return, one example
        i = len(closes) // 2
        meanlog = float(np.mean(np.log(closes[i - lb + 1:i + 1])))
        lookback_ret = float(np.log(closes[i]) - np.log(closes[i - lb]))
        print(f"{ticker} lb={lb}: +1={npos} -1={nneg} 0={nzero} flips={flips} "
              f"meanlog={meanlog:+.3f} lb_ret={lookback_ret:+.4f}")
    print()

# turnover / cost impact for AMZN lookback 5: zero vs realistic
bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)
zero = bt.run_bars(list(bars), sig, bt.BacktestConfig())
real = bt.BacktestConfig(initial_capital=1e6, commission_per_trade=2.0,
                         commission_per_share=0.003, slippage_cents=2.0,
                         slippage_proportional=0.0005)
res_real = bt.run_bars(list(bars), sig, real)
gross_turnover = sum(abs(f.shares) for f in res_real.trades)
total_comm = sum(f.commission for f in res_real.trades)
print("AMZN lb5 full-sample: n_trades=", len(res_real.trades),
      "gross_traded_shares=", round(gross_turnover, 1),
      "total_commission=", round(total_comm, 2))
print("  zero-cost total_return=", round((zero.equity_curve[-1]-1e6)/1e6, 4),
      " realistic total_return=", round((res_real.equity_curve[-1]-1e6)/1e6, 4))
print("  n_fills zero=", len(zero.trades), " n_fills realistic=", len(res_real.trades))