import sys, json
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt
from research.checks.verify_momentum_cost_sensitivity import (
    vol_blocks, segments_from_labels, momentum_weights, cfg_for, null_median, TRAIN, TEST,
    WARM, OVERLAP, WINDOW, N_BLOCKS, MIN_SEGMENT_BARS)

art = json.load(open("state/check_artifacts/momentum_cost_sensitivity_results.json"))
for ticker in ("AMZN", "JPM"):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    for lb in (3, 5, 10):
        for bps in (0.0, 10.0, 30.0):
            med, nulls, nfs = [], [], []
            for _lab, s, e in runs:
                seg_bars = list(bars)[s:e]
                seg_closes = np.array([b.close for b in seg_bars], dtype=np.float64)
                w = momentum_weights(seg_closes, lb)
                sig = [bt.Signal(date=i + 1, weight=w[i]) for i in range(len(seg_bars))]
                res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                                      warmup=WARM, overlap_window=OVERLAP, cfg=cfg_for(bps))
                med.append(round(float(np.median(np.log1p(np.clip(
                    [f.metrics["total_return"] for f in res.folds], -1.0 + 1e-12, None)))), 3))
                nulls.append(round(null_median(seg_bars, lb, bps), 3))
                nfs.append(len(res.folds))
            pub = art["per_asset"][ticker][str(lb)]["cost_bps"][str(bps)]
            ok = med == pub["medians"] and nulls == pub["null_medians"] and nfs[0] == pub["n_folds"]
            print(ticker, lb, bps, "MATCH" if ok else "MISMATCH", med, nulls, nfs[0],
                  "" if ok else pub)