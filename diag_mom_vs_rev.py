"""Compare momentum vs reversal fold structures on the same segments."""
import json

rev = json.load(open("state/check_artifacts/mean_reversion_results.json"))
mom = json.load(open("state/check_artifacts/momentum_results.json"))

print("reversal verdict counts:", rev["universe"]["verdict_counts"])
print("momentum verdict counts:", mom["universe"]["verdict_counts"])

for t in rev["universe"]["per_asset"]:
    rv = rev["universe"]["per_asset"][t]
    mv = mom["universe"]["per_asset"][t]
    print("{:8s} rev_med={:28s} mom_med={:28s} rev_v={:18s} mom_v={:18s}"
          .format(t, str(rv["medians"]), str(mv["medians"]), rv["verdict"], mv["verdict"]))

# Inspect fold geometry: recompute walk-forward fold counts on both signals
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt
import numpy as np

def segments(closes, n_blocks, window, min_seg):
    labels = bt.volatility_blocks(closes, n_blocks=n_blocks, window=window)
    runs = []
    cur = labels[0]; rs = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - rs >= min_seg:
                runs.append((cur, rs, i))
            cur = labels[i]; rs = i
    if len(labels) - rs >= min_seg:
        runs.append((cur, rs, len(labels)))
    return runs

def fold_count(ticker, sig_fn, train, test, warm, overlap):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    runs = segments(closes, 4, 60, 400)
    total = 0
    for _, s, e in runs:
        seg_bars = list(bars)[s:e]
        seg_signals = sig_fn(closes)[s:e]
        res = bt.walk_forward(seg_bars, seg_signals, train_window=train,
                              test_window=test, warmup=warm,
                              overlap_window=overlap,
                              cfg=bt.BacktestConfig(warmup_periods=warm))
        total += len(res.folds)
    return len(runs), total, runs

def fold_medians(ticker, sig_fn, train, test, warm, overlap):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    runs = segments(closes, 4, 60, 400)
    meds = []
    for lab, s, e in runs:
        seg_bars = list(bars)[s:e]
        seg_signals = sig_fn(closes)[s:e]
        res = bt.walk_forward(seg_bars, seg_signals, train_window=train,
                              test_window=test, warmup=warm,
                              overlap_window=overlap,
                              cfg=bt.BacktestConfig(warmup_periods=warm))
        lr = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                       for f in res.folds], dtype=np.float64)
        meds.append(round(float(np.median(lr)), 3))
    return meds

for ticker in ["AAPL", "AMZN", "JPM"]:
    print("=== {} ===".format(ticker))
    for name, sig in [("momentum", lambda c: bt.momentum_signals(c, 5)),
                      ("reversal", lambda c: bt.mean_reversion_signals(c, 5))]:
        nseg, nf, runs = fold_count(ticker, sig, 252, 84, 60, 60)
        meds = fold_medians(ticker, sig, 252, 84, 60, 60)
        print("  {} ({} folds over {} segments): {}".format(name, nf, nseg, meds))
