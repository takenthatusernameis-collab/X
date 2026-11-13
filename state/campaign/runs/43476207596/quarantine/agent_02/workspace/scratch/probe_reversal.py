"""Diagnostic: confirm mean_reversion_signals and compare to corrected logic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def main():
    for ticker in ["AAPL", "AMZN"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        for lb in (3, 5, 10):
            rev = bt.momentum_signals(closes, lookback=lb)  # placeholder to keep parity
            rev = bt.mean_reversion_signals(closes, lookback=lb)
            w = np.array([s.weight for s in rev if s.weight != 0.0])
            print(f"{ticker} reversal lb={lb}: nonneutral={len(w)} "
                  f"long={int(np.sum(w>0))} short={int(np.sum(w<0))}")
        # corrected return-based momentum
        lb = 5
        corr = []
        for i in range(lb, len(closes)):
            r = float(np.log(closes[i]) - np.log(closes[i - lb]))
            corr.append(1.0 if r > 0 else -1.0)
        print(f"{ticker} corrected momentum lb={lb}: long={corr.count(1.0)} "
              f"short={corr.count(-1.0)} flips={sum(1 for a,b in zip(corr,corr[1:]) if a!=b)}")


if __name__ == "__main__":
    main()