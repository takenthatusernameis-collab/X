"""Diagnostic probe: verify what research.backtest.momentum_signals computes.

The docstring claims "long the previous lookback-day return". This probe checks
whether the implemented signal actually tracks the sign of the previous
lookback-day RETURN, or something else (e.g. the sign of the mean log PRICE).
Read-only diagnostic; writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def probe(ticker: str, lookback: int = 5) -> None:
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=lookback)
    weights = np.array([s.weight for s in sig])
    nonzero = weights[weights != 0.0]
    n_long = int(np.sum(nonzero > 0))
    n_short = int(np.sum(nonzero < 0))
    print(f"{ticker}: bars={len(closes)} nonneutral={len(nonzero)} "
          f"long={n_long} short={n_short} "
          f"price_range=[{closes.min():.3f},{closes.max():.3f}] "
          f"log_price_min={np.log(closes.min()):.3f}")

    # What the implementation effectively computes:
    i = len(closes) - 1
    impl_ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
    # What the docstring claims: sign of previous lookback-day return.
    true_ret = float(np.log(closes[i]) - np.log(closes[i - lookback]))
    print(f"  last bar impl_metric(mean log price)={impl_ret:+.4f} -> weight "
          f"{1.0 if impl_ret > 0 else -1.0:+.0f}; "
          f"true prev {lookback}d return={true_ret:+.4f} -> would be "
          f"{1.0 if true_ret > 0 else -1.0:+.0f}")

    # Fraction of bars where implementation sign == true return sign.
    signs_impl = []
    signs_true = []
    for j in range(lookback, len(closes)):
        signs_impl.append(1.0 if np.mean(np.log(closes[j - lookback + 1 : j + 1])) > 0 else -1.0)
        signs_true.append(1.0 if (np.log(closes[j]) - np.log(closes[j - lookback])) > 0 else -1.0)
    agree = float(np.mean(np.array(signs_impl) == np.array(signs_true)))
    print(f"  sign agreement impl vs true return: {agree:.3f}")


if __name__ == "__main__":
    for t in ["AAPL", "AMZN", "JPM", "XOM"]:
        probe(t, 5)
    print("\nDone.")