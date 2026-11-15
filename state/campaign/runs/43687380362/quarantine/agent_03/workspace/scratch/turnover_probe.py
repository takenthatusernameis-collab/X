import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

cfg = bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                        slippage_cents=2.0, slippage_proportional=0.0005)
print("ticker  n_bars  long   short  neutral  flips  traded_notional  commission  net_cost")
for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]:
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    w = np.array([s.weight for s in sig])
    n_long = int(np.sum(w > 0))
    n_short = int(np.sum(w < 0))
    n_neutral = int(np.sum(w == 0))
    flips = int(np.sum(np.diff(w) != 0))
    res = bt.run_bars(list(bars), sig, cfg)
    traded = sum(abs(f.shares) * closes[f.date - 1] for f in res.trades)
    comm = sum(f.commission for f in res.trades)
    slip_impact = sum(-f.shares * (f.price - closes[f.date - 1]) for f in res.trades)
    print("{:6s} {:6d} {:6d} {:6d} {:8d} {:6d} {:16.0f} {:11.0f} {:9.0f}".format(
        ticker, len(closes), n_long, n_short, n_neutral, flips, traded, comm, slip_impact))