import sys, json
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt
from research.checks.verify_momentum_cost_sensitivity import (
    vol_blocks, segments_from_labels, momentum_weights, cfg_for, null_median, TRAIN, TEST,
    WARM, OVERLAP, WINDOW, N_BLOCKS, MIN_SEGMENT_BARS)

art = json.load(open("state/check_artifacts/momentum_cost_sensitivity_results.json"))
bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
print("runs", runs)
s, e = runs[0][1], runs[0][2]
seg_bars = list(bars)[s:e]
w = momentum_weights(closes, 5)
sig = [bt.Signal(date=i + 1, weight=w[i]) for i in range(s, e)]
res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                      warmup=WARM, overlap_window=OVERLAP, cfg=cfg_for(0.0))
print("n folds mine", len(res.folds), "seg len", e - s)
med = float(np.median([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in res.folds]))
print("median mine", repr(med), "rounded", round(med, 3))
print("published", art["per_asset"]["AMZN"]["5"]["cost_bps"]["0.0"]["n_folds"],
      art["per_asset"]["AMZN"]["5"]["cost_bps"]["0.0"]["medians"])
print("null mine", repr(null_median(seg_bars, 5, 0.0)))
print("null published", art["per_asset"]["AMZN"]["5"]["cost_bps"]["0.0"]["null_medians"])