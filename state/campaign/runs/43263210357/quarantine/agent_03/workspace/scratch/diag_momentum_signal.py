"""Read-only diagnostic: is momentum_signals actually a momentum signal?

The docstring of research/backtest/regime_stability.py::momentum_signals says
'long the previous lookback-day return ... if the last lookback days were up,
go long'. The implementation computes:

    ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))

which is the mean of LOG PRICES, not the mean of LOG RETURNS. If every adjusted
close is above 1.0 the mean log price is always positive, so the signal would be
constant +1 (always long) regardless of lookback. This script tests that
directly on the collected universe and compares against a correct past-only
momentum signal (sign of closes[i] - closes[i-lookback]).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def correct_momentum_signals(closes, lookback=5):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        up = float(closes[i]) > float(closes[i - lookback])
        out[i] = bt.Signal(date=i + 1, weight=1.0 if up else -1.0)
    return out


def weight_counts(sig, start):
    vals = [s.weight for s in sig[start:]]
    return {
        "n": len(vals),
        "long": sum(1 for v in vals if v > 0),
        "short": sum(1 for v in vals if v < 0),
        "neutral": sum(1 for v in vals if v == 0),
    }


def main():
    manifest = bt.load_universe()
    print("=== momentum_signals weight composition (lookback=5, after warmup) ===")
    print(f"{'ticker':6s} {'bars':>6s} {'existing long/short':>22s} {'correct long/short':>22s}")
    for ticker, (bars, _dates) in manifest.items():
        closes = bars.closes_array()
        ex = weight_counts(bt.momentum_signals(closes, lookback=5), 5)
        co = weight_counts(correct_momentum_signals(closes, lookback=5), 5)
        print(f"{ticker:6s} {len(closes):6d} "
              f"{ex['long']:8d}/{ex['short']:<8d}    "
              f"{co['long']:8d}/{co['short']:<8d}")

    # Confirm the existing signal is identical to always-long on a sample.
    closes = manifest["AAPL"][0].closes_array()
    ex = bt.momentum_signals(closes, lookback=5)
    al = bt.always_long_signal(closes, 5, 5)
    same = all(a.weight == b.weight for a, b in zip(ex[5:], al[5:]))
    print(f"\nAAPL existing momentum == always_long (bars 5..end): {same}")

    # Quantify the edge difference on AAPL, zero cost, one walk-forward window.
    print("\n=== AAPL walk-forward median log total return (zero cost) ===")
    bars = list(manifest["AAPL"][0])
    for name, sig in (
        ("existing momentum(5)  [mean log PRICE]", bt.momentum_signals(closes, 5)),
        ("correct  momentum(5)  [sign chg]", correct_momentum_signals(closes, 5)),
        ("always_long           [reference]", al),
    ):
        res = bt.walk_forward(bars, sig, train_window=252, test_window=84,
                              warmup=60, overlap_window=60)
        print(f"  {name:42s} median_log={res.aggregate_metrics['median_log_total_return']:+.4f} "
              f"mean_log={res.aggregate_metrics['mean_log_total_return']:+.4f} "
              f"pos_folds={res.aggregate_metrics['fold_count_positive']}/{res.aggregate_metrics['n_folds']}")

    # Does the mean-of-log-prices equal the mean-of-log-returns? Show a few values.
    print("\n=== raw signal value check (AAPL, i=100..104) ===")
    for i in range(100, 105):
        price_based = float(np.mean(np.log(closes[i - 5 + 1:i + 1])))
        return_based = float(np.log(closes[i]) - np.log(closes[i - 5]))
        print(f"  i={i}: price_mean={price_based:+.4f}  true_ret={return_based:+.4f}  "
              f"sign(price_mean)>0={price_based > 0}  sign(true_ret)>0={return_based > 0}")


if __name__ == "__main__":
    main()