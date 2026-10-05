import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

tick = sys.argv[1]
train, test, warm, overlap = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEGMENT_BARS = 60, 4, 400
LOOKBACK = 5

bars, dates = bt.load_ticker(tick)
closes = bars.closes_array()

def vol_blocks_plus1(closes, n_blocks, window):
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(float(np.std(np.log(closes[i - window : i + 1]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(np.median([v for v in bar_vols[s:e] if v > 0]) if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels

def seg_from_labels(labels, min_segment_bars):
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs

fw_labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
my_labels = vol_blocks_plus1(closes, N_BLOCKS, WINDOW)

print("tick:", tick)
print("n bars:", len(closes))
print("framework n_runs:", len(seg_from_labels(fw_labels, MIN_SEGMENT_BARS)))
print("plus1     n_runs:", len(seg_from_labels(my_labels, MIN_SEGMENT_BARS)))
print("labels differ:", fw_labels != my_labels)
if fw_labels != my_labels:
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(fw_labels, my_labels)) if a != b]
    print("first 20 diff positions:", diffs[:20])
    # show labels with indices around first diff
    d = diffs[0][0]
    print("framework :", fw_labels[max(0,d-8):d+8])
    print("plus1     :", my_labels[max(0,d-8):d+8])

# segment medians for each
for name, labels in [("framework", fw_labels), ("plus1", my_labels)]:
    runs = seg_from_labels(labels, MIN_SEGMENT_BARS)
    print("\n{} segments: {}".format(name, [(lab, s, e) for lab, s, e in runs]))
    for lab, s, e in runs:
        seg_bars = list(bars)[s:e]
        seg_signals = bt.momentum_signals(closes, LOOKBACK)[s:e]
        cfg = bt.BacktestConfig(warmup_periods=warm)
        res = bt.walk_forward(seg_bars, seg_signals, train_window=train, test_window=test, warmup=warm, overlap_window=overlap, cfg=cfg)
        log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in res.folds], dtype=np.float64)
        print("  {}: {} -> med {:.4f}".format(lab, (s, e), float(np.median(log))))
