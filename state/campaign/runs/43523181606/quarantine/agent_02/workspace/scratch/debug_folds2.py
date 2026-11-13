import sys, json
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt
from research.checks import verify_momentum_cost_sensitivity as V

bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
runs = V.segments_from_labels(V.vol_blocks(closes, V.N_BLOCKS, V.WINDOW), V.MIN_SEGMENT_BARS)
s, e = runs[0][1], runs[0][2]
seg_bars = list(bars)[s:e]
print("s,e", s, e, "len", len(seg_bars))
seg_closes = np.array([b.close for b in seg_bars], dtype=np.float64)
w = V.momentum_weights(seg_closes, 3)
sig = [bt.Signal(date=i + 1, weight=w[i]) for i in range(len(seg_bars))]
print("len sig", len(sig))
res = bt.walk_forward(seg_bars, sig, train_window=V.TRAIN, test_window=V.TEST,
                      warmup=V.WARM, overlap_window=V.OVERLAP, cfg=V.cfg_for(0.0))
print("inline folds", len(res.folds))
print("via function", V.segment_result("AMZN", bars, s, e, 3, 0.0)[2])
import inspect
print(inspect.getsource(V.segment_result))