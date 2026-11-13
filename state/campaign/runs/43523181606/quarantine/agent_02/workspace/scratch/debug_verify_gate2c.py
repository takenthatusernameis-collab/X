import sys, json
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt
from research.checks.verify_momentum_cost_sensitivity import (
    vol_blocks, segments_from_labels, segment_result, WINDOW, N_BLOCKS, MIN_SEGMENT_BARS, LOOKBACKS, COSTS_BPS)

art = json.load(open("state/check_artifacts/momentum_cost_sensitivity_results.json"))
bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
pub = art["per_asset"]["AMZN"]["3"]["cost_bps"]["0.0"]
med, nulls, nf = [], [], None
for _lab, s, e in runs:
    m, n, nf = segment_result("AMZN", bars, s, e, 3, 0.0)
    med.append(round(m, 3)); nulls.append(round(n, 3))
print("med", med, [type(x) for x in med])
print("pub med", pub["medians"], [type(x) for x in pub["medians"]])
print("eq", med == pub["medians"])
print("nulls eq", nulls == pub["null_medians"])
print("segs", [r[0] for r in runs], pub["segments"], [r[0] for r in runs] == pub["segments"])
print("nf", nf, type(nf), pub["n_folds"], type(pub["n_folds"]), nf == pub["n_folds"])
print("keys", list(pub.keys()))