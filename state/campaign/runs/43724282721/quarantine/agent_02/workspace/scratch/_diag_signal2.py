"""Confirm repo momentum medians equal a pure always-long backtest."""
import sys

sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt


def always_long_fn(closes, **params):
    return [bt.Signal(date=i + 1, weight=1.0) for i in range(len(closes))]


def med(signals_fn, closes, bars, lookback, baseline):
    labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = signals_fn(closes, **{k: v for k, v in baseline})
    r = bt.stress_segments(
        signals_fn=signals_fn, bars=bars, signals=sig,
        param_grid=[dict(baseline)], baseline=tuple(baseline),
        segment_fn=seg_fn, train_window=252, test_window=84, warmup=60,
        overlap_window=60, periods_per_year=252, min_segment_bars=400)
    return [round(s.baseline_median_log_return, 4) for s in r.scenarios], r.overall_verdict


for t in ["AAPL", "AMZN", "TSLA", "NVDA"]:
    bars, _ = bt.load_ticker(t)
    closes = bars.closes_array()
    repo, v1 = med(bt.momentum_signals, closes, bars, 5, (("lookback", 5),))
    al, v2 = med(always_long_fn, closes, bars, 5, (("lookback", 5),))
    print("{:6s} repo={} ({}) always_long={} ({}) identical={}".format(
        t, repo, v1, al, v2, repo == al and v1 == v2))

# Also confirm the *true* momentum signal is NOT static (turnover) and that the
# repo momentum's single sign change is just the initial entry.
for t in ["AAPL"]:
    bars, _ = bt.load_ticker(t)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    first_long = next(i for i, s in enumerate(sig) if s.weight > 0)
    print("AAPL first long bar index:", first_long, "of", len(sig),
          "| last short bar:", max([i for i, s in enumerate(sig) if s.weight < 0] or [-1]))