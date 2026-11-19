"""Diagnostic: is the momentum signal actually lookback-dependent?

Counts sign agreement between momentum signals at lookbacks 3, 5, 10 on the
collected tickers. If the signals were lookback-invariant, agreement would be
100%.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def main():
    for ticker in ["AAPL", "AMZN", "TSLA"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        sigs = {lb: bt.momentum_signals(closes, lookback=lb) for lb in (3, 5, 10)}
        w = {lb: np.array([s.weight for s in sigs[lb]]) for lb in (3, 5, 10)}
        for a, b in ((3, 5), (5, 10), (3, 10)):
            both = (w[a] != 0) & (w[b] != 0)
            agree = float(np.mean(w[a][both] == w[b][both]))
            print(f"{ticker}: lb{a} vs lb{b}: nonzero_both={int(both.sum())} "
                  f"sign_agreement={agree:.4f} diff_days={int((w[a] != w[b]).sum())}")
        # Also compute total-return sign directly to confirm formula.
        for lb in (3, 5, 10):
            direct = np.sign(closes[lb:] - closes[:-lb])
            impl = w[lb][lb:]
            print(f"{ticker}: lb{lb} impl-vs-direct sign mismatch = "
                  f"{int((direct != impl).sum())}")


if __name__ == "__main__":
    main()