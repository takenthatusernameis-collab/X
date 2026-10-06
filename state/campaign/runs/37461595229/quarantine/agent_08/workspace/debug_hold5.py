import sys
sys.path.insert(0, '.')
import numpy as np
import research.backtest as bt
from research.data.preflight import load_manifest

manifest = load_manifest()
for ticker in ["AMZN", "JPM"]:
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, 5)
    n = len(closes)
    hold = 5
    eff = [bt.Signal(date=i+1, weight=0.0) for i in range(n)]
    for i in range(5, n):
        idx = i - (i - 5) % hold
        eff[i] = bt.Signal(date=i+1, weight=sig[idx].weight)
    labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
    # framework-style segmentation
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
    print(f"{ticker}: segments={[(l,s,e) for l,s,e in runs]}, n_bars={n}")
    # walk-forward on segment slice
    seg_medians = []
    for lab, s, e in runs:
        seg_bars = list(bars)[s:e]
        seg_signals = eff[s:e]
        cfg = bt.BacktestConfig(warmup_periods=60)
        res = bt.walk_forward(
            seg_bars, seg_signals, train_window=252, test_window=84,
            warmup=60, overlap_window=60, cfg=cfg)
        m = float(np.median([np.log1p(np.clip(f.metrics["total_return"], -1+1e-12, None)) for f in res.folds]))
        seg_medians.append(round(m, 3))
        print(f"  seg {lab} [{s}:{e}] folds={len(res.folds)} median={m:.6f} -> {round(m,3)}")
    print(f"  median: {seg_medians}")
    print(f"  published: {'[0.005, -0.101, 0.055]' if ticker=='AMZN' else '[-0.053, 0.021, 0.045]'}")
