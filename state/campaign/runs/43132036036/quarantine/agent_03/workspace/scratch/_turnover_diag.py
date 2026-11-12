"""Quantify turnover: implemented momentum vs matched coin-flip null."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

REALISTIC = dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005)


def main() -> int:
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    w = np.array([s.weight for s in sig])
    print("momentum full-sample position changes:", int(np.sum(w[1:] != w[:-1])))
    for name, cfg in [("zero", bt.BacktestConfig()), ("realistic", bt.BacktestConfig(**REALISTIC))]:
        res = bt.run_bars(list(bars), sig, cfg)
        print("{}: full-sample fills={} final_equity={:.1f}".format(name, len(res.trades), res.equity_curve[-1]))
    # null turnover
    null = bt.random_signals(len(closes), seed=42)
    nw = np.array([s.weight for s in null])
    print("null position changes:", int(np.sum(nw[1:] != nw[:-1])))
    for name, cfg in [("zero", bt.BacktestConfig()), ("realistic", bt.BacktestConfig(**REALISTIC))]:
        res = bt.run_bars(list(bars), null, cfg)
        print("null {}: fills={} final_equity={:.1f}".format(name, len(res.trades), res.equity_curve[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())