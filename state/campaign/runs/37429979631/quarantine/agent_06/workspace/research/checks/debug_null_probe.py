"""Direct probe: compare null-signal generation approaches against artifact."""
import sys, json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.engine import Signal

ARTIFACT = json.load(open(Path.cwd() / "state" / "check_artifacts"
                            / "momentum_cost_sensitivity_results.json"))
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60

def vol_blocks(closes, n_blocks, window):
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = [0.0 if i < window else
                float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0)
                for i in range(n)]
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bm = float(np.median([v for v in bar_vols[s:e] if v > 0])
                       if e - s > window else 0.0)
            labels.append("turbulent" if bm > series_med else "calm")
    return labels


def segs(labels, m):
    runs = []
    cur, rs = labels[0], 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - rs >= m:
                runs.append((cur, rs, i))
            cur, rs = labels[i], i
    if len(labels) - rs >= m:
        runs.append((cur, rs, len(labels)))
    return runs


def rand(n, seed):
    rng = np.random.default_rng(seed)
    w = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0/3, 1.0/3, 1.0/3])
    return [Signal(date=i+1, weight=float(x)) for i, x in enumerate(w)]


bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
labels = vol_blocks(closes, 4, 60)
runs = segs(labels, 400)
print("AMZN segments:", [(l, s, e, e-s) for l, s, e in runs])
seed = 42 + int(round(5.0) * 1000)
print("seed:", seed)

pub_null = ARTIFACT["per_asset"]["AMZN"]["momentum"]["5"]["zero"]["null_medians"]
print("artifact null medians:", pub_null)

seg = list(bars)[runs[0][1]:runs[0][2]]
sig_full = rand(len(closes), seed)
sig_seg = rand(len(seg), seed)
r1 = bt.walk_forward(seg, sig_full[runs[0][1]:runs[0][2]],
                     train_window=TRAIN, test_window=TEST, warmup=0,
                     overlap_window=0, cfg=bt.BacktestConfig())
r2 = bt.walk_forward(seg, sig_seg,
                     train_window=TRAIN, test_window=TEST, warmup=0,
                     overlap_window=0, cfg=bt.BacktestConfig())
m1 = [np.log1p(np.clip(f.metrics["total_return"], -1+1e-12, None)) for f in r1.folds]
m2 = [np.log1p(np.clip(f.metrics["total_return"], -1+1e-12, None)) for f in r2.folds]
m3 = [np.log1p(np.clip(f.metrics["total_return"], -1+1e-12, None)) for f in r1.folds]

print("methodA (full-signal, slice) folds:", len(m1), "median:", round(float(np.median(m1)), 3))
print("methodB (seg-len signal) folds:", len(m2), "median:", round(float(np.median(m2)), 3))

print("first 6 weights sig_full[s0:e0]:", [round(float(x), 2) for x in sig_full[runs[0][1]:runs[0][1]+6]])
print("first 6 weights sig_seg:", [round(float(x), 2) for x in sig_seg[:6]])

# Also recompute the check's exact path: noise_benchmark via parameter_sweep
bars_full, _ = bt.load_ticker("AMZN")
seg2 = list(bars_full)[runs[0][1]:runs[0][2]]
noise = bt.noise_benchmark(seg2, param_grid=[{"lookback": 5.0}],
                           train_window=TRAIN, test_window=TEST,
                           warmup=0, overlap_window=0,
                           cfg=bt.BacktestConfig(), periods_per_year=252)
print("noise_benchmark baseline median:", round(noise.baseline_median_log_return, 3))
