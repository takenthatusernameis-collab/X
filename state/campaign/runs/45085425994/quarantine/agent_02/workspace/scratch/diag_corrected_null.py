"""Diagnostic: corrected causal momentum vs matched coin-flip null, with costs.

For each asset we compare the corrected momentum candidate medians against a
coin-flip sign null *that trades the same number of times* (a random-sign
version of the same signal), under each cost point. The candidate-minus-null gap
is the decision-relevant quantity; the raw level is not.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400
SEED = 42
COSTS = {
    "zero": dict(),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "stress2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
                     slippage_cents=4.0, slippage_proportional=0.0010),
}


def momentum_causal(closes, lookback=5):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(int(lookback), n):
        ret = float(closes[i] / closes[i - int(lookback)] - 1.0)
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def random_sign(n, seed):
    rng = np.random.default_rng(seed)
    return [bt.Signal(date=i + 1, weight=float(w))
            for i, w in enumerate(rng.choice([-1.0, 1.0], size=n))]


def seg_medians(bars, sig, cost, min_segment_bars=MIN_SEG):
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    runs = []
    cur, start = labels[0], 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                runs.append((cur, start, i))
            cur, start = labels[i], i
    if len(labels) - start >= min_segment_bars:
        runs.append((cur, start, len(labels)))

    cfg = bt.BacktestConfig(initial_capital=1e6, **COSTS[cost])
    out = []
    bar_list = list(bars)
    for lab, s, e in runs:
        sb = bar_list[s:e]
        ss = sig[s:e]
        res = bt.walk_forward(sb, ss, train_window=TRAIN, test_window=TEST,
                              warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
        lr = np.array([np.log1p(np.clip(f.metrics["total_return"], -1 + 1e-12, None))
                       for f in res.folds])
        out.append((lab, round(float(np.median(lr)), 3), len(res.folds)))
    return out


manifest = __import__("json").loads(
    (Path(__file__).resolve().parent.parent / "research" / "data" / "manifest.json").read_text())

print("=== corrected momentum candidate vs matched random-sign null (lookback 5) ===")
print("  gap = candidate median - null median (same cost model, same segments)")
print()
for cost in COSTS:
    print(f"-- cost point: {cost}")
    gaps = []
    for t in manifest["universe"]:
        bars = bt.load_ticker(t)[0]
        closes = bars.closes_array()
        cand = seg_medians(bars, momentum_causal(closes, 5), cost)
        null = seg_medians(bars, random_sign(len(closes), SEED), cost)
        cvals = [c[1] for c in cand]
        nvals = [n[1] for n in null]
        g = round(float(np.median(cvals) - np.median(nvals)), 3)
        gaps.append(g)
        print("  {:6s} cand={} null={} gap={:+.3f} segs={}".format(
            t, cvals, nvals, g, [c[0] for c in cand]))
    print(f"  median gap across assets: {round(float(np.median(gaps)), 3)}"
          f"  positive_assets={sum(1 for g in gaps if g > 0)}/10")
    print()