"""Probe: AMZN segment structure and walk-forward fold counts under
candidate (warmup=60, overlap=60) vs null (warmup=0, overlap=0) settings."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400


def volatility_blocks(closes, n_blocks, window):
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        bar_vols.append(0.0 if i < window else
                        float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            block = i // block_size
            s, e = block * block_size, (block + 1) * block_size
            bmed = float(np.median([v for v in bar_vols[s:e] if v > 0])
                         if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_segment_bars):
    runs = []
    cur, run_start = labels[0], 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur, run_start = labels[i], i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs


bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
print("AMZN n_bars:", len(closes))
labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
print("segments (label, start, end, len):")
for lab, s, e in runs:
    print("  {} [{:4d}:{:4d}] len={}".format(lab, s, e, e - s))
print("total segment bars:", sum(e - s for _, s, e in runs))

# Walk-forward fold counts per segment under both settings
for label, (s, e) in [(r[0], (r[1], r[2])) for r in runs]:
    seg = list(bars)[s:e]
    for warm, ov in (("warmup=60, overlap=60", WARM, OVERLAP),
                     ("warmup=0, overlap=0", 0, 0)):
        res = bt.walk_forward(seg, [bt.Signal(date=i + 1, weight=0.0) for i in range(len(seg))],
                              train_window=TRAIN, test_window=TEST, warmup=warm,
                              overlap_window=ov, cfg=bt.BacktestConfig())
        print("  {} folds={}".format(warm, res.aggregate_metrics["n_folds"]))
