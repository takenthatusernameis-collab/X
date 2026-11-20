import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.checks.momentum_cost_sensitivity import cost_config, REALISTIC_COST

for ticker in ("AMZN", "AAPL", "JPM"):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    print(ticker)
    for mult in (0.0, 1.0, 2.0, 4.0):
        res = bt.run_bars(list(bars), sig, cost_config(mult))
        eq = res.equity_curve
        total_return = (eq[-1] - 1e6) / 1e6
        comm = sum(f.commission for f in res.trades)
        abs_shares = sum(abs(f.shares) for f in res.trades)
        gross = sum(abs(f.shares) * f.price for f in res.trades)
        print("  mult={:.1f} n_trades={} total_comm={:+.1f} gross_traded={:.3e} "
              "sum|delta_shares|={:.3e} total_return={:+.4f}".format(
                  mult, len(res.trades), comm, gross, abs_shares, total_return))