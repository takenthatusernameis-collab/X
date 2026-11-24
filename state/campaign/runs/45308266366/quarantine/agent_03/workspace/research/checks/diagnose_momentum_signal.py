"""Diagnostic: verify what `momentum_signals` actually computes.

Motivation: the momentum artifacts (`momentum_results.json`,
`momentum_lookback_sweep_results.json`) show *identical* per-segment candidate
medians across lookbacks 3/5/10 for every asset (e.g. AAPL
[0.071, 0.054, 0.092, 0.069] at all three lookbacks). A genuine
"long the previous N-day return" signal must vary with N. This diagnostic
tests whether `momentum_signals` is lookback-sensitive at all.

It reports, per ticker and lookback:
- the set of distinct signal weights (a genuine momentum signal takes both
  +1 and -1);
- the fraction of bars that are long (a genuine momentum signal is ~0.5, an
  always-long signal is 1.0);
- the fraction of bars whose sign matches a *correct* momentum signal
  (sign of the lookback log return), where chance is ~0.5 and a correct
  implementation is ~1.0;
- the daily turnover of the existing signal vs a correct momentum signal,
  which is what a transaction-cost test is sensitive to.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "JPM", "XOM"]
LOOKBACKS = (3, 5, 10)


def correct_momentum_signals(closes, lookback):
    """A genuine momentum signal: +1 if the previous `lookback`-day log
    return is positive, else -1 (0 before the window)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lg = np.log(closes)
    for i in range(lookback, n):
        r = lg[i] - lg[i - lookback]
        out[i] = bt.Signal(date=i + 1, weight=1.0 if r > 0 else -1.0)
    return out


def turnover(signals):
    """Mean absolute daily change in position weight (a proxy for the
    per-bar trading intensity a cost model charges on)."""
    w = np.array([s.weight for s in signals], dtype=np.float64)
    return float(np.mean(np.abs(np.diff(w))))


def main() -> int:
    print("=== momentum_signals degeneracy diagnostic ===")
    print("For each ticker/lookback: distinct weights, frac_long, "
          "frac_sign_match_correct, turnover(existing) vs turnover(correct)\n")
    header = "{:6s} {:>4s}  {:22s} {:>8s} {:>10s}  {:>10s} {:>10s}".format(
        "ticker", "lb", "distinct_weights", "frac_long",
        "sign_match", "turn_exst", "turn_corr")
    print(header)
    print("-" * len(header))
    for ticker in TICKERS:
        bars, _dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        for lb in LOOKBACKS:
            sig = bt.momentum_signals(closes, lookback=lb)
            w = np.array([s.weight for s in sig], dtype=np.float64)
            distinct = sorted(set(np.round(w[lb:], 6).tolist()))
            frac_long = float(np.mean(w[lb:] > 0))
            corr = correct_momentum_signals(closes, lb)
            cw = np.array([s.weight for s in corr], dtype=np.float64)
            # sign agreement on active bars (both non-neutral)
            active = (w[lb:] != 0) & (cw[lb:] != 0)
            sign_match = (float(np.mean(np.sign(w[lb:][active]) ==
                                        np.sign(cw[lb:][active])))
                          if active.any() else float("nan"))
            t_exst = turnover(sig)
            t_corr = turnover(corr)
            print("{:6s} {:>4d}  {:22s} {:8.3f} {:10.3f}  {:10.4f} {:10.4f}".format(
                ticker, lb, str(distinct), frac_long, sign_match, t_exst, t_corr))
        # price range for context
        print("       {} close range [{:.3f}, {:.3f}]  min log price {:.3f}".format(
            ticker, float(closes.min()), float(closes.max()),
            float(np.log(closes).min())))
    print()
    print("Interpretation:")
    print(" - distinct_weights == [1.0] and frac_long == 1.000 => the signal is")
    print("   always long and does NOT depend on the lookback (not momentum).")
    print(" - sign_match ~ 0.5 => the existing signal agrees with a genuine")
    print("   momentum signal at chance rate.")
    print(" - turn_exst << turn_corr => a cost test on the existing signal barely")
    print("   charges turnover, so it cannot measure momentum cost robustness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())