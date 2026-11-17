"""Diagnostic: inspect the as-implemented momentum signal.

Checks whether `bt.momentum_signals` (and `bt.mean_reversion_signals`) produce a
timing signal that flips with the sign of the recent RETURN, or a near-static
position. The implementation computes
    ret = mean(log(closes[i-lookback+1 : i+1]))
which is the mean log PRICE over the window, not a return. If closes are all
above 1, that quantity is positive at essentially every bar, so the signal is a
static long (momentum) or static short (reversal).

Research / simulation only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt


def main() -> int:
    for ticker in ["AAPL", "AMZN", "JPM"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        for lookback in (3, 5, 10):
            sig = bt.momentum_signals(closes, lookback=lookback)
            w = np.array([s.weight for s in sig])
            n_active = int(np.sum(w != 0.0))
            n_pos = int(np.sum(w > 0))
            n_neg = int(np.sum(w < 0))
            changes = int(np.sum(w[1:] != w[:-1]))
            # Intended momentum: sign of the trailing lookback simple return.
            intended = np.zeros(len(closes))
            for i in range(lookback, len(closes)):
                r = closes[i] / closes[i - lookback] - 1.0
                intended[i] = 1.0 if r > 0 else -1.0
            intended_changes = int(np.sum(intended[1:] != intended[:-1]))
            agree = float(np.mean(np.sign(w[lookback:]) == np.sign(intended[lookback:])))
            print(f"{ticker} lookback={lookback}: active={n_active} pos={n_pos} "
                  f"neg={n_neg} changes={changes} | intended_changes="
                  f"{intended_changes} sign_agreement={agree:.3f}")
        print()

    # Show the raw quantity the implementation thresholds.
    bars, _ = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    lb = 5
    q = [float(np.mean(np.log(closes[i - lb + 1: i + 1]))) for i in range(lb, len(closes))]
    print("AAPL mean-log-price quantity (lookback 5): min={:.4f} max={:.4f} "
          "frac_positive={:.3f}".format(min(q), max(q), np.mean(np.array(q) > 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())