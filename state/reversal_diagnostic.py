#!/usr/bin/env python3
"""Independent diagnostic: replicate the check's stress_segments on AMZN using
walk_forward directly on segment slices, with signals recomputed from raw CSV.

This isolates the discrepancy: the check's mean-reversion AMZN segment medians
[-0.202, -0.193, -0.128] vs an independent recomputation."""
import csv, math
import numpy as np
from pathlib import Path

sys = __import__("sys")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest.engine import walk_forward, run_bars

data = Path("/home/runner/work/X/X/research/data/raw/AMZN_daily.csv")
closes = []
dates = []
with open(data) as f:
    r = csv.DictReader(f)
    for row in r:
        closes.append(float(row["adjclose"]))
        dates.append(row["date"])
closes = np.array(closes, dtype=np.float64)
n = len(closes)
print(f"AMZN closes: {n} bars, {dates[0]}..{dates[-1]}\n")

def make_bars(closes):
    return [bt.Bar(date=d, open=c, high=c, low=c, close=c, volume=0.0)
            for c, d in zip(closes, dates)]

def ma_signal(closes, fast=20, slow=60):
    n = len(closes)
    out = [bt.Signal(date=i+1, weight=0.0) for i in range(n)]
    for i in range(slow-1, n):
        fm = float(np.mean(closes[i-fast+1:i+1]))
        sm = float(np.mean(closes[i-slow+1:i+1]))
        out[i] = bt.Signal(date=i+1, weight=1.0 if fm > sm else -1.0)
    return out

def rev_signal_check_style(closes, lookback=5):
    """Exact copy of research.backtest.mean_reversion_signals (mean of daily log returns)."""
    n = len(closes)
    out = [bt.Signal(date=i+1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i-lookback+1 : i+1])))
        out[i] = bt.Signal(date=i+1, weight=-1.0 if ret > 0 else 1.0)
    return out


def rev_signal_my_style(closes, lookback=5):
    """My independent version (log of total lookback return)."""
    n = len(closes)
    out = [bt.Signal(date=i+1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = np.log(closes[i]) - np.log(closes[i-lookback])
        out[i] = bt.Signal(date=i+1, weight=-1.0 if ret > 0 else 1.0)
    return out

def vol_blocks(closes, n_blocks=4, window=60):
    """Same algorithm as research.backtest.volatility_blocks."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(float(np.std(np.log(closes[i-window:i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b+1) * block_size
            bmed = float(np.median([v for v in bar_vols[s:e] if v > 0]) if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels

def segments_from_labels(labels, min_segment_bars=400):
    runs = []
    cur = labels[0]; run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]; run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs

bars = make_bars(closes)
labels = vol_blocks(closes)
runs = segments_from_labels(labels)
print(f"Segments: {[(lab, s, e) for lab, s, e in runs]}")

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60

def seg_median(seg_bars, seg_signals):
    res = walk_forward(seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
                       warmup=WARM, overlap_window=OVERLAP)
    m = np.array([math.log1p(np.clip(f.metrics["total_return"], -1+1e-12, None)) for f in res.folds])
    return round(float(np.median(m)), 3), len(res.folds)

# 1) Compare the two reversal signal implementations bar by bar.
sig_check = rev_signal_check_style(closes)
sig_my = rev_signal_my_style(closes)
diffs = sum(1 for a, b in zip(sig_check, sig_my) if a.weight != b.weight)
print(f"\nSignal-level comparison (check-style vs my-style): {n} bars, {diffs} weight disagreements")
ds = [i for i in range(n) if sig_check[i].weight != sig_my[i].weight][:10]
print(f"  first disagreement indices: {ds}")
for i in ds[:5]:
    r_check = np.mean(np.log(closes[i-5+1:i+1]))
    r_my = np.log(closes[i]) - np.log(closes[i-5])
    print(f"  bar {i}: check_style mean_daily_logret={r_check:+.6f} sig={sig_check[i].weight:+.1f} | "
          f"my_style total_logret={r_my:+.6f} sig={sig_my[i].weight:+.1f}")

for name, sig_fn in [("MA(20/60)", ma_signal),
                     ("REVERSAL (check-style)", rev_signal_check_style),
                     ("REVERSAL (my-style)", rev_signal_my_style)]:
    sig = sig_fn(closes)
    print(f"\n{name}:")
    for lab, s, e in runs:
        med, nf = seg_median(list(bars)[s:e], sig[s:e])
        print(f"  {lab:9s} bars={e-s} folds={nf} median_log={med:+.3f}")

print("\nCheck artifact published (mean_reversion AMZN): base_ma [0.046, 0.001, -0.154], "
      "mean_reversion [-0.202, -0.193, -0.128]")
