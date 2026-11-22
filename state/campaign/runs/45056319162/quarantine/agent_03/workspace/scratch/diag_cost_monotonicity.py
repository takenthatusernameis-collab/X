"""Diagnostic: why does the cost-matched coin-flip null move non-monotonically
with the cost multiplier? Reproduce one segment's null folds at several cost
levels from a single fixed coin-flip signal and print per-fold returns.

Research/simulation only. No live trading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest.perturbation import random_signals

WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
TRAIN, TEST = 252, 84
REALISTIC = dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005)


def vol_blocks(closes, n_blocks, window):
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0))
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
    return labels


def segments_from_labels(labels, min_bars):
    runs = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_bars:
                runs.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_bars:
        runs.append((cur, start, len(labels)))
    return runs


def cfg_for(mult):
    return bt.BacktestConfig(**{k: v * mult for k, v in REALISTIC.items()})


def fold_returns(seg_bars, sig, mult, warmup=0, overlap=0):
    res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                          warmup=warmup, overlap_window=overlap, cfg=cfg_for(mult))
    return [float(f.metrics["total_return"]) for f in res.folds]


def main():
    ticker = "NVDA"
    lookback = 5
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    print(f"{ticker} segments: {[(lab, e - s) for lab, s, e in runs]}")
    # first segment (the one that shows +1.134 at m=0.5)
    lab, s, e = runs[0]
    seg_bars = list(bars)[s:e]
    n = len(seg_bars)
    seed = 42 + lookback * 1000  # noise_benchmark _param_seed for {"lookback":5}
    sig = random_signals(n, seed=seed)
    print(f"segment {lab} len={n} null seed={seed}")
    print(f"signal weight counts: "
          f"{ {w: sum(1 for x in sig if x.weight == w) for w in (-1.0, 0.0, 1.0)} }")
    for mult in (0.0, 0.5, 1.0, 2.0):
        r = fold_returns(seg_bars, sig, mult)
        lr = np.log1p(np.clip(r, -1.0 + 1e-12, None))
        print(f"  m={mult}: n_folds={len(r)} median_log={np.median(lr):+.4f} "
              f"folds={[round(x, 3) for x in r]}")
    print()
    # Commission-only vs slippage-only decomposition (lookback 5, this segment).
    variants = {
        "commission_only": dict(commission_per_trade=2.0, commission_per_share=0.003,
                                slippage_cents=0.0, slippage_proportional=0.0),
        "slippage_only": dict(commission_per_trade=0.0, commission_per_share=0.0,
                              slippage_cents=2.0, slippage_proportional=0.0005),
        "both_realistic": dict(REALISTIC),
    }
    for name, costs in variants.items():
        res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                              warmup=0, overlap_window=0, cfg=bt.BacktestConfig(**costs))
        r = [float(f.metrics["total_return"]) for f in res.folds]
        lr = np.log1p(np.clip(r, -1.0 + 1e-12, None))
        print(f"  {name}: median_log={np.median(lr):+.4f}")


if __name__ == "__main__":
    main()