"""Diagnostic probe: is research.backtest.momentum_signals a genuine
short-horizon momentum signal, or a degenerate (near-constant) exposure?

No modification to the research path. Read-only inspection.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest


def main():
    m = load_manifest()
    print("ticker  n_bars  min_close  max_close  frac_logmean_pos  "
          "n_weight+1  n_weight-1  n_weight0  n_sign_flips")
    for entry in m["entries"]:
        t = entry["ticker"]
        bars, dates = bt.load_ticker(t)
        closes = bars.closes_array()
        lb = 5
        n = len(closes)
        # replicate the exact expression used in momentum_signals
        meanlog = np.array([np.mean(np.log(closes[i - lb + 1:i + 1]))
                            for i in range(n)])
        frac_pos = float(np.mean(meanlog > 0))
        sig = bt.momentum_signals(closes, lookback=lb)
        w = np.array([s.weight for s in sig])
        n_pos = int(np.sum(w > 0))
        n_neg = int(np.sum(w < 0))
        n_zero = int(np.sum(w == 0))
        flips = int(np.sum(np.diff(w[n_zero:]) != 0)) if n_zero < n else 0
        print("{:6s}  {:5d}  {:9.4f}  {:9.4f}  {:16.4f}  {:10d}  "
              "{:10d}  {:9d}  {:10d}".format(
                  t, n, float(np.min(closes)), float(np.max(closes)),
                  frac_pos, n_pos, n_neg, n_zero, flips))

    # Show the actual log-return expression that a genuine momentum signal
    # would use, for comparison, on AMZN.
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    lb = 5
    genuine = np.sign([np.log(closes[i]) - np.log(closes[i - lb])
                       for i in range(lb, len(closes))])
    print("\nAMZN genuine log-return-sign momentum sign distribution: "
          "frac_plus1={:.4f} frac_minus1={:.4f}".format(
              float(np.mean(genuine > 0)), float(np.mean(genuine < 0))))
    buggy = np.sign([np.mean(np.log(closes[i - lb + 1:i + 1]))
                     for i in range(lb, len(closes))])
    print("AMZN implemented momentum_signals sign distribution: "
          "frac_plus1={:.4f} frac_minus1={:.4f}".format(
              float(np.mean(buggy > 0)), float(np.mean(buggy < 0))))
    return 0


if __name__ == "__main__":
    sys.exit(main())