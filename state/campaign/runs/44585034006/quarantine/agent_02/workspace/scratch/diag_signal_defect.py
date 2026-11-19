"""Diagnostic: quantify the momentum/reversal signal defect.

For each collected ticker, report:
- fraction of post-warmup momentum weights equal to +1 / -1 / 0
- fraction of post-warmup reversal weights equal to -1 / +1 / 0
- the geometric-mean-price rule that the implementation actually applies
- a corrected momentum signal (sign of cumulative log return) for contrast
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def corrected_momentum(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        r = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if r > 0 else -1.0)
    return out


def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
               "JPM", "JNJ", "XOM"]
    for t in tickers:
        bars, _ = bt.load_ticker(t)
        closes = bars.closes_array()
        lb = 5
        mom = np.array([s.weight for s in bt.momentum_signals(closes, lb)])
        rev = np.array([s.weight for s in bt.mean_reversion_signals(closes, lb)])
        cor = np.array([s.weight for s in corrected_momentum(closes, lb)])
        nz = mom != 0
        print(f"{t}: mom +1={np.mean(mom[nz] > 0):.3f} -1={np.mean(mom[nz] < 0):.3f} "
              f"| rev -1={np.mean(rev[nz] < 0):.3f} +1={np.mean(rev[nz] > 0):.3f} "
              f"| corrected +1={np.mean(cor[nz] > 0):.3f} -1={np.mean(cor[nz] < 0):.3f} "
              f"| price range [{closes.min():.2f},{closes.max():.2f}]")
    # Show that the implementation equals sign(mean(log(price))) exactly.
    bars, _ = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    lb = 5
    impl = np.array([s.weight for s in bt.momentum_signals(closes, lb)])
    rule = np.sign(np.convolve(np.log(closes), np.ones(lb) / lb, mode="full")[:len(closes)])
    print("AAPL impl == sign(mean log price over window):",
          bool(np.all(impl[lb:] == rule[lb:])))


if __name__ == "__main__":
    main()