import sys, json
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt
from research.checks.verify_momentum_cost_sensitivity import (
    vol_blocks, segments_from_labels, momentum_weights, cfg_for, WINDOW, N_BLOCKS,
    MIN_SEGMENT_BARS, TRAIN, TEST, WARM, OVERLAP)
from research.checks import verify_momentum_cost_sensitivity as V

print("module consts", TRAIN, TEST, WARM, OVERLAP)
bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
s, e = runs[0][1], runs[0][2]
seg_bars = list(bars)[s:e]
print("seg len", len(seg_bars), "runs", runs)
for warm in (0, 60):
    res = bt.walk_forward(seg_bars, [bt.Signal(date=i + 1, weight=0.0) for i in range(len(seg_bars))],
                          train_window=TRAIN, test_window=TEST, warmup=warm,
                          overlap_window=OVERLAP, cfg=cfg_for(0.0))
    print("warmup", warm, "folds", len(res.folds))

res2 = bt.stress_segments(
    signals_fn=bt.momentum_signals,
    bars=bars,
    signals=bt.momentum_signals(closes, lookback=3),
    param_grid=[{"lookback": 3}],
    baseline=(("lookback", 3),),
    segment_fn=bt.segment_fn_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW)),
    train_window=TRAIN, test_window=TEST, warmup=WARM, overlap_window=OVERLAP,
    cfg=cfg_for(0.0), periods_per_year=252, min_segment_bars=MIN_SEGMENT_BARS,
)
print("stress_segments n_folds", res2.n_folds, "scenarios", len(res2.scenarios))
print("seg lens from labels", [(r[0], r[2] - r[1]) for r in runs])