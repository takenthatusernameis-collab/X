"""Read-only probe: how similar are momentum signals across lookbacks?

The lookback-sweep artifact reports identical per-segment medians for lookbacks
3/5/10 (and the longer-horizon check reported the same for 20/60). This probe
quantifies signal agreement so the cost-sensitivity artifact can be interpreted
correctly: a sign-only signal can be nearly constant across lookbacks on a
drifting asset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

LOOKBACKS = (3, 5, 10, 20, 60)


def main() -> int:
    for ticker in ("AAPL", "AMZN", "JPM", "JNJ"):
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        sig = {
            lb: np.array([s.weight for s in bt.momentum_signals(closes, lb)])
            for lb in LOOKBACKS
        }
        print(ticker, "n_bars", len(closes))
        for lb in LOOKBACKS:
            w = sig[lb]
            print("  lookback {:2d}: long_share={:.3f} short_share={:.3f} neutral={}".format(
                lb, float(np.mean(w > 0)), float(np.mean(w < 0)),
                int(np.sum(w == 0))))
        for a, b in ((3, 5), (3, 10), (5, 10), (5, 20), (20, 60)):
            print("  agree lookback {} vs {}: {:.4f}".format(
                a, b, float(np.mean(sig[a] == sig[b]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())