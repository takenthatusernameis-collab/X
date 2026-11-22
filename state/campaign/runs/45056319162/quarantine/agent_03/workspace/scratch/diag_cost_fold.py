"""Diagnostic: inspect one null fold's equity path at m=0 vs m=0.5 to locate
the source of the >100% fold returns produced by a fixed coin-flip signal once
a small cost is added.

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
    bs = n // n_blocks
    bv = []
    for i in range(n):
        bv.append(0.0 if i < window else
                  float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bv if v > 0]
    med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // bs
            s, e = b * bs, (b + 1) * bs
            bmed = float(np.median([v for v in bv[s:e] if v > 0]) if e - s > window else 0.0)
            labels.append("turbulent" if bmed > med else "calm")
    return labels


def segments_from_labels(labels, mb):
    runs, cur, st = [], labels[0], 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - st >= mb:
                runs.append((cur, st, i))
            cur, st = labels[i], i
    if len(labels) - st >= mb:
        runs.append((cur, st, len(labels)))
    return runs


def cfg(mult):
    return bt.BacktestConfig(**{k: v * mult for k, v in REALISTIC.items()})


def describe(tag, bars, sig, mult):
    # replicate walk_forward's first fold construction
    n = len(bars)
    warmup, overlap = 0, 0
    fold_end = warmup + TRAIN + TEST
    fs = 0
    fold_bars = bars[fs:fold_end]
    fold_signals = sig[fs:fold_end]
    fold_cfg = bt.BacktestConfig(
        initial_capital=1e6, target_exposure=1.0,
        commission_per_trade=REALISTIC["commission_per_trade"] * mult,
        commission_per_share=REALISTIC["commission_per_share"] * mult,
        slippage_cents=REALISTIC["slippage_cents"] * mult,
        slippage_proportional=REALISTIC["slippage_proportional"] * mult,
        warmup_periods=warmup)
    res = bt.Backtest(fold_cfg).run(fold_bars, fold_signals)
    eq = res.equity_curve
    test_eq = eq[-TEST:]
    total_return = test_eq[-1] / test_eq[0] - 1.0
    cum_shares = sum(f.shares for f in res.trades)
    final_cash = eq[-1] - cum_shares * fold_bars[-1].close
    print(f"[{tag}] m={mult}: OOS start={test_eq[0]:.1f} end={test_eq[-1]:.1f} "
          f"total_return={total_return:+.3f}")
    print(f"    eq min={eq.min():.1f} max={eq.max():.1f} final_cash={final_cash:.1f} "
          f"final_shares={cum_shares:.1f} n_fills={len(res.trades)} "
          f"sum_comm={sum(f.commission for f in res.trades):.1f}")
    # largest single-bar equity jumps and the signal weight on that bar
    d = np.diff(eq)
    idx = np.argsort(-np.abs(d))[:5]
    for i in sorted(idx):
        w = fold_signals[i + 1].weight
        print(f"    bar {i + 1}: close={fold_bars[i + 1].close:.2f} weight={w:+.0f} "
              f"d_equity={d[i]:+.1f} eq={eq[i + 1]:.1f}")


def main():
    bars, _ = bt.load_ticker("NVDA")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    lab, s, e = runs[0]
    seg_bars = list(bars)[s:e]
    n = len(seg_bars)
    sig = random_signals(n, seed=42 + 5 * 1000)
    print(f"NVDA segment {lab} len={n}")
    describe("m=0.0", seg_bars, sig, 0.0)
    describe("m=0.5", seg_bars, sig, 0.5)
    # show the closes range in the OOS window
    oos = seg_bars[TRAIN:TRAIN + TEST]
    print(f"OOS closes: first={oos[0].close:.2f} last={oos[-1].close:.2f} "
          f"min={min(b.close for b in oos):.2f} max={max(b.close for b in oos):.2f}")


if __name__ == "__main__":
    main()