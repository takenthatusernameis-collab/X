import sys
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt

LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400


def hold_effective_signals(signals, hold):
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < LOOKBACK:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - LOOKBACK) % hold
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out


bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=LOOKBACK)
labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
runs = []
cur = labels[0]; start = 0
for i in range(1, len(labels)):
    if labels[i] != cur:
        if i - start >= MIN_SEG:
            runs.append((cur, start, i))
        cur = labels[i]; start = i
if len(labels) - start >= MIN_SEG:
    runs.append((cur, start, len(labels)))
print("segments:", [(n, s, e) for n, s, e in runs])

for hold in (1, 2, 3, 5):
    eff = hold_effective_signals(sig, hold)
    w = [s.weight for s in eff]
    changes = sum(1 for i in range(1, len(w)) if w[i] != w[i - 1])
    print(f"hold={hold}: n_weight_changes={changes} first20={[int(x) for x in w[5:25]]}")

for name, s, e in runs:
    seg_bars = list(bars)[s:e]
    print(f"--- segment {name} [{s}:{e}] ---")
    for hold in (1, 2, 3, 5):
        eff = hold_effective_signals(sig, hold)
        seg_sig = eff[s:e]
        res = bt.walk_forward(seg_bars, seg_sig, train_window=TRAIN, test_window=TEST,
                              warmup=WARM, overlap_window=OVERLAP, cfg=bt.BacktestConfig())
        rets = [f.metrics["total_return"] for f in res.folds]
        med = float(np.median(np.log1p(np.clip(rets, -1 + 1e-12, None))))
        print(f"  hold={hold}: n_folds={len(res.folds)} median_log={med:+.6f}")