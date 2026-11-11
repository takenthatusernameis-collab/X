"""Diagnostic: what series is the momentum signal actually betting on?

`research/backtest/regime_stability.py::momentum_signals` computes

    ret = mean(log(closes[i - lookback + 1 : i + 1]))

which is the mean LOG PRICE over the window, not a return. For any price
series with all closes > 1 the value is strictly positive, so the emitted
weight is +1 at every bar after the warmup window and the "momentum" bet is a
constant long position. This diagnostic quantifies that for the collected
universe: weight composition, sign changes, and realized turnover under the
repository's realistic cost model.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

LOOKBACKS = (3, 5, 10)
TICKERS = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM")


def weights(closes, lookback):
    n = len(closes)
    w = np.zeros(n, dtype=np.float64)
    for i in range(int(lookback), n):
        ret = float(np.mean(np.log(closes[i - int(lookback) + 1: i + 1])))
        w[i] = 1.0 if ret > 0 else -1.0
    return w


def main() -> int:
    cfg = bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )
    print("ticker lookback  n_long  n_short  sign_changes  turnover/yr  min_window_mean_log_price")
    for ticker in TICKERS:
        bars = bt.load_ticker(ticker)[0]
        closes = bars.closes_array()
        for lb in LOOKBACKS:
            w = weights(closes, lb)
            n_long = int(np.sum(w > 0))
            n_short = int(np.sum(w < 0))
            changes = int(np.sum(np.diff(np.sign(w)) != 0))
            sig = [bt.Signal(date=i + 1, weight=float(w[i])) for i in range(len(w))]
            res = bt.run_bars(list(bars), sig, cfg)
            notional = float(sum(abs(f.shares) * f.price for f in res.trades))
            years = bars.n_bars / 252.0
            turn = notional / 1e6 / years
            min_mean = min(
                float(np.mean(np.log(closes[i - lb + 1: i + 1])))
                for i in range(lb, len(closes))
            )
            print(f"{ticker:6s} {lb:<8d} {n_long:<7d} {n_short:<8d} {changes:<13d} "
                  f"{turn:<12.4f} {min_mean:+.3f}")

    print("\nInterpretation: min_window_mean_log_price > 0 for every cell means the")
    print("window mean log price can never be negative, so the emitted weight is")
    print("constant and the series is a buy-and-hold position, not a momentum bet.")
    print("Research/simulation only; no live trading or production execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())