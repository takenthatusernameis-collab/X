"""Verify what bt.momentum_signals / mean_reversion_signals actually compute.

The docstring says "long the previous N-day RETURN". The code computes
mean(log(close)) over the window, which is the mean log PRICE, not a return.
For any price > 1 this is always positive, so the signal is constant +1
(momentum) / -1 (reversal) regardless of lookback. Verify quantitatively.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]


def sign_changes(sig):
    return sum(1 for i in range(1, len(sig))
               if abs(sig[i].weight - sig[i - 1].weight) > 1e-12)


def true_momentum(closes, lookback):
    """Textbook past-only momentum: sign of the trailing lookback return."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        past_ret = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if past_ret > 0 else -1.0)
    return out


print("ticker  lb  repo_frac_long  repo_sign_changes  true_frac_long  true_sign_changes")
for t in TICKERS:
    bars, _ = bt.load_ticker(t)
    closes = bars.closes_array()
    for lb in (3, 5, 10):
        repo = bt.momentum_signals(closes, lookback=lb)
        repo_long = np.mean([s.weight > 0 for s in repo])
        true = true_momentum(closes, lookback=lb)
        true_long = np.mean([s.weight > 0 for s in true])
        print("{:6s} {:2d}  {:14.4f}  {:16d}  {:13.4f}  {:16d}".format(
            t, lb, repo_long, sign_changes(repo), true_long, sign_changes(true)))

# Confirm: repo momentum medians == pure always-long backtest medians.
print("\nConfirm repo momentum == always-long (lookback 5):")
for t in ["AAPL", "NVDA", "TSLA"]:
    bars, _ = bt.load_ticker(t)
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
    seg_fn = bt.segment_fn_from_labels(labels)
    always_long = [bt.Signal(date=i + 1, weight=1.0) for i in range(len(closes))]
    repo_sig = bt.momentum_signals(closes, lookback=5)

    def med(sig):
        r = bt.stress_segments(
            signals_fn=lambda c, **p: sig, bars=bars, signals=sig,
            param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
            segment_fn=seg_fn, train_window=252, test_window=84, warmup=60,
            overlap_window=60, periods_per_year=252, min_segment_bars=400)
        return [round(s.baseline_median_log_return, 4) for s in r.scenarios]

    print("  {:6s} repo={} always_long={} identical={}".format(
        t, med(repo_sig), med(always_long), med(repo_sig) == med(always_long)))