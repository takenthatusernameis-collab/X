import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

artifact = json.loads((Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json").read_text())
pub = artifact["per_asset"]["AMZN"]["costs"]["zero"]["3"]
print("published n_folds", pub["n_folds"])
bars, _ = bt.load_ticker("AMZN")
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
# copy segments from verifier
runs = []
cur = labels[0]
start = 0
for i in range(1, len(labels)):
    if labels[i] != cur:
        if i - start >= 400:
            runs.append((cur, start, i))
        cur = labels[i]
        start = i
if len(labels) - start >= 400:
    runs.append((cur, start, len(labels)))
for s, e in [(s, e) for _, s, e in runs]:
    seg_bars = list(bars)[s:e]
    seg_closes = np.array([float(b.close) for b in seg_bars], dtype=np.float64)
    sig = []
    lb = 3
    for i in range(lb, len(seg_closes)):
        ret = float(np.log(seg_closes[i] / seg_closes[i - lb]))
        sig.append(bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0))
    sig = [bt.Signal(date=i + 1, weight=0.0) for i in range(lb)] + sig
    res = bt.walk_forward(seg_bars, sig, train_window=252, test_window=84, warmup=60, overlap_window=60, cfg=bt.BacktestConfig())
    print(s, e, len(res.folds), len(seg_bars))