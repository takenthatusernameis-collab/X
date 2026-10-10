#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/runner/work/X/X')
import research.backtest as bt
import numpy as np

# Test segment_median function from verification script
from research.data.preflight import load_manifest

# Helper function from verification script
def segment_median(ticker, s, e, train, test, warm, overlap, cfg, signals_fn):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = signals_fn(closes, **{"fast": 20, "slow": 60}) if signals_fn.__name__ == "base_ma_signals" else signals_fn(closes)
    if len(signals) != len(closes):
        raise ValueError(f"{ticker}: signals {len(signals)} != bars {len(closes)}")
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)

# Test the verification script's base_ma_signals
fast, slow = 20, 60

# Load manifest and get bars
manifest = load_manifest()
tickers = {}
for entry in manifest["entries"]:
    tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

bars, dates = bt.load_ticker("AMZN")
closes = bars.closes_array()
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            block = i // block_size
            s, e = block * block_size, (block + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window
                else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels

def segments_from_labels(labels, min_segment_bars):
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs

def momentum_signals(closes, lookback):
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out

# Custom base_ma_signals from verification script (lines 108-117)
def base_ma_signals(closes, fast, slow):
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    fast, slow = int(fast), int(slow)
    for i in range(max(fast, slow) - 1, n):
        fast_ma = float(np.mean(closes[i - fast + 1 : i + 1]))
        slow_ma = float(np.mean(closes[i - slow + 1 : i + 1]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if fast_ma > slow_ma else -1.0)
    return out

# Get labels and runs
labels = vol_blocks(closes, N_BLOCKS, WINDOW)
runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
print(f"Number of segments: {len(runs)}")

# Compute base MA medians using verification script's implementation
base_medians = []
for lab, s, e in runs:
    cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=60)
    m, nf = segment_median("AMZN", s, e, 252, 84, 60, 60, cfg, base_ma_signals)
    base_medians.append(round(m, 3))

print(f"Verification script base MA medians: {base_medians}")
print(f"Verification script base MA labels: {[lab for lab, _, _ in runs]}")

# Now compute using framework's ma_crossover_signals
base_medians_framework = []
for lab, s, e in runs:
    cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=60)
    m, nf = segment_median("AMZN", s, e, 252, 84, 60, 60, cfg, bt.ma_crossover_signals)
    base_medians_framework.append(round(m, 3))

print(f"Framework ma_crossover_signals medians: {base_medians_framework}")
print(f"Framework ma_crossover_signals labels: {[lab for lab, _, _ in runs]}")

# Compare
print(f"\nMatch: {base_medians == base_medians_framework}")
print(f"Labels same: {[lab for lab, _, _ in runs]} == {[lab for lab, _, _ in runs]}")