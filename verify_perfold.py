import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import research.backtest as bt

# tiny family, 60 bars
rng = np.random.default_rng(0)
fam = []
prices = {a: np.array([100.0 + rng.random() * 10 for _ in range(60)], dtype=np.float64) for a in range(3)}
for a in range(3):
    dates = np.arange(1, 61, dtype=np.int64)
    c = prices[a]; h = np.maximum(c, np.roll(c, -1)); l = np.minimum(c, np.roll(c, -1))
    fam.append(bt.BarSequence(dates, c, h, l, c, np.full(60, 1e6)))

pnl = bt.csrs_spread_family(fam, 2, 1, 1)
synth = bt.build_synthetic_spread_asset(pnl, start_price=1e6)

TRAIN, TEST, WARM, OVERLAP = 10, 5, 3, 2
STEP = TEST - OVERLAP

# --- direct path (mirror walk_forward_fold_log_returns) ---
def direct(pnl, train, test, warm, overlap):
    n = len(pnl); step = test - overlap; fold_start = 0; lrets = []
    while True:
        fold_end = fold_start + warm + train + test
        if fold_end > n: break
        oos = pnl[fold_end - test + 1:fold_end]
        comp = 1.0
        for r in oos: comp *= 1.0 + r
        lrets.append(float(np.log1p(np.clip(comp - 1.0, -1.0 + 1e-12, None))))
        fold_start += step
    return lrets

# --- engine path per fold with weight baseline = fold's first post-warmup bar ---
def engine(pnl, train, test, warm, overlap):
    closes = synth.closes_array()
    lrets = []
    fold_start = 0
    step = test - overlap
    while True:
        fold_end = fold_start + warm + train + test
        if fold_end > len(pnl): break
        seg_bars = list(synth)[fold_start:fold_end]
        seg_closes = closes[fold_start:fold_end]
        seg_signals = [bt.Signal(date=seg_closes[k], weight=closes[fold_start + k] / closes[fold_start + warm])
                       for k in range(len(seg_closes))]
        res = bt.walk_forward(seg_bars, seg_signals, train_window=train, test_window=test,
                              warmup=warm, overlap_window=overlap,
                              cfg=bt.BacktestConfig(periods_per_year=252))
        lr = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in res.folds]
        lrets.extend(lr)
        fold_start += step
    return lrets

d = direct(pnl, TRAIN, TEST, WARM, OVERLAP)
e = engine(pnl, TRAIN, TEST, WARM, OVERLAP)
print("direct:", [round(x, 9) for x in d])
print("engine:", [round(x, 9) for x in e])
print("match:", np.allclose(d, e, atol=1e-9))
