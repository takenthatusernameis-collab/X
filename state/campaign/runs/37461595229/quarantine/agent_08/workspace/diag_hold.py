import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
manifest = load_manifest()
m = next(e for e in manifest["entries"] if e["ticker"] == "AMZN")
bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()

def momentum_signals(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out

def hold_effective(signals, hold):
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < 5:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - 5) % hold
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out

def medians_for(signals, block_size):
    """Approx: medians per volatility block, using 4 blocks of equal size,
    walk-forward 252/84/60/60, min 400 bars/segment."""
    labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
    from research.backtest.regime_stability import segment_fn_from_labels
    seg_fn = segment_fn_from_labels(labels)
    from research.backtest.metrics import compute_metrics
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= 400:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= 400:
        runs.append((cur, run_start, len(labels)))
    segs = []
    for lab, s, e in runs:
        seg_bars = list(bars)[s:e]
        seg_sig = signals[s:e]
        cfg = bt.BacktestConfig(warmup_periods=60)
        res = bt.walk_forward(
            seg_bars, seg_sig, train_window=252, test_window=84,
            warmup=60, overlap_window=60, cfg=cfg)
        m = float(np.median([np.log1p(np.clip(f.metrics["total_return"], -1+1e-12, None)) for f in res.folds]))
        segs.append((lab, m, len(res.folds)))
    return segs

sig = momentum_signals(closes, 5)
# count sign flips vs hold-2 borrow
diff_h2 = sum(1 for i in range(5, len(sig)) if sig[i].weight != sig[i-1].weight)
diff_h5 = sum(1 for i in range(5, len(sig)) if sig[i].weight != sig[i-4].weight)
print("AMZN momentum sign flips vs previous bar:", diff_h2)
print("AMZN momentum sign flips vs bar-4:", diff_h5)

for hold in (1, 2, 3, 5):
    eff = hold_effective(sig, hold)
    segs = medians_for(eff, 4)
    ms = [round(m, 4) for _, m, _ in segs]
    print(f"hold={hold}: {segs} medians={ms}")

# full sample equity
cfg = bt.BacktestConfig(initial_capital=1e6)
for hold in (1, 2, 5):
    eff = hold_effective(sig, hold)
    res = bt.run_bars(list(bars), eff, cfg)
    print(f"hold={hold}: total_return={(res.equity_curve[-1]-1e6)/1e6:.6f}, trades={len(res.trades)}")
