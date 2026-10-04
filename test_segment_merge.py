import sys
import numpy as np
sys.path.insert(0, "/home/runner/work/X/X")
from collections import namedtuple
import research.backtest as bt
from research.data.preflight import load_manifest

BarRun = namedtuple("BarRun", ["label", "start", "end"])

def run_length_encode(labels):
    runs = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            runs.append(BarRun(cur, start, i))
            cur = labels[i]
            start = i
    runs.append(BarRun(cur, start, len(labels)))
    return runs

def merge_short_runs(runs, min_run_bars):
    """Absorb runs shorter than min_run_bars into the larger adjacent run.

    Runs merge into the larger of the two neighbors; if only one neighbor
    exists it merges there; if none, the run is kept as-is.
    """
    merged = []
    i = 0
    while i < len(runs):
        r = runs[i]
        if r.end - r.start >= min_run_bars:
            merged.append(r)
            i += 1
        else:
            prev = merged[-1] if merged else None
            nxt = runs[i + 1] if i + 1 < len(runs) else None
            if prev and nxt:
                if prev.end - prev.start >= nxt.end - nxt.start:
                    merged[-1] = BarRun(prev.label, prev.start, r.end)
                else:
                    merged.append(BarRun(nxt.label, r.start, nxt.end))
                    del runs[i + 1]
                    i -= 1
                    continue
            elif prev:
                merged[-1] = BarRun(prev.label, prev.start, r.end)
            elif nxt:
                merged.append(BarRun(nxt.label, r.start, nxt.end))
                del runs[i + 1]
            i += 1
    return merged

manifest = load_manifest()
bars, _ = bt.load_ticker("AAPL")
closes = bars.closes_array()
n = len(closes)

# per-bar regime by trailing-60d vol vs series median
vols = [float(np.std(np.log(closes[i - 60:i]), ddof=1)) * np.sqrt(252.0)
        for i in range(60, n)]
med = float(np.median(vols))
labels = ["insufficient"] * 60 + (
    ["low" if v < med else "high" for v in vols]
)

runs = run_length_encode(labels)
print("raw runs:", len(runs), "longest:", max(r.end - r.start for r in runs))

for min_run in (10, 20, 40):
    merged = merge_short_runs(runs, min_run)
    kept = [r for r in merged if r.end - r.start >= 400]
    print(f"\nmin_run_bars={min_run}: merged={len(merged)} segments")
    for r in kept:
        print(f"  {r.label}: {r.start}..{r.end} ({r.end-r.start} bars, {r.end-r.start/n:.1%})")
    dropped = [f"{r.label}:{r.end-r.start}" for r in merged if r.end - r.start < 400]
    print(f"  dropped runs <400: {dropped}")
