"""Diagnose the null-median mismatch between the check and the verifier.

Hypothesis: `stress_segments` computes its null via `noise_benchmark(..., warmup=0,
overlap_window=0)` on the segment, whereas my verifier used warmup=WARM. The
coin-flip signal is generated from `n = len(bars)`, so a different warmup changes
which fold windows exist and therefore the null median.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
RAW = ROOT / "research" / "data" / "raw"


def read_adjclose(ticker):
    vals = []
    with open(RAW / f"{ticker}_daily.csv") as fh:
        fh.readline()
        for line in fh:
            if line.strip():
                vals.append(float(line.split(",")[5]))
    return np.asarray(vals, dtype=np.float64)


closes = read_adjclose("XOM")
# segment from earlier output: XOM calm/turbulent, use first two runs by date blocks
n = len(closes)
window, n_blocks = 60, 4
block_size = n // n_blocks
bar_vols = [0.0] * window + [
    float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0)
    for i in range(window, n)]
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
runs = []
cur, start = labels[0], 0
for i in range(1, len(labels)):
    if labels[i] != cur:
        if i - start >= 400:
            runs.append((cur, start, i))
        cur, start = labels[i], i
if len(labels) - start >= 400:
    runs.append((cur, start, len(labels)))
print("runs:", runs)

for cost, kw in [("realistic", dict(commission_per_trade=2.0, commission_per_share=0.003,
                                    slippage_cents=2.0, slippage_proportional=0.0005))]:
    for lab, s, e in runs:
        seg = [bt.Bar(date=i + 1, open=closes[i], high=closes[i], low=closes[i],
                      close=closes[i], volume=0.0) for i in range(s, e)]
        a = noise_benchmark(bars=seg, param_grid=[{"lookback": 3}],
                            train_window=TRAIN, test_window=TEST,
                            warmup=0, overlap_window=0, periods_per_year=252,
                            cfg=bt.BacktestConfig(initial_capital=1e6, **kw))
        b = noise_benchmark(bars=seg, param_grid=[{"lookback": 3}],
                            train_window=TRAIN, test_window=TEST,
                            warmup=WARM, overlap_window=OVERLAP, periods_per_year=252,
                            cfg=bt.BacktestConfig(initial_capital=1e6, **kw))
        print(f"  seg {lab} [{s}:{e}] warmup0={round(a.baseline_median_log_return,3)} "
              f"warmup60={round(b.baseline_median_log_return,3)} "
              f"n_folds0={a.n_folds} n_folds60={b.n_folds}")