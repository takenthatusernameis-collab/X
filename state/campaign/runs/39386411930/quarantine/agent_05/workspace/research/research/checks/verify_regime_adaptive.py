"""Independent verification of the regime-adaptive MA crossover results
(`research/checks/regime_adaptive_ma.py`).

This is a separate implementation path: it recomputes each segment's walk-
forward median and the regime-adaptive variant medians using `walk_forward`
directly (not ``stress_segments``), then compares against the published
adaptive figures. A mismatch would flag a defect in the adaptive check.

Published adaptive figures (seed 42, MA windows: calm 20/60, turbulent 10/30,
train=252d/test=84d, warmup=60d, overlap=60d, min_segment_bars=400):

  AMZN medians [base]:      [0.046, +0.001, -0.154]   (turbulent, calm, turbulent)
  AMZN medians [adaptive]:  [-0.028, +0.001, +0.015]
  AMZN medians [turb_only]: [0.046, +0.000, -0.154]
  JPM  medians [base]:      [0.109, -0.006, +0.033]
  JPM  medians [adaptive]:  [0.069, -0.006, -0.001]
  JPM  medians [turb_only]: [0.109, +0.000, +0.033]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
FAST, SLOW = 20, 60
FAST_TURB, SLOW_TURB = 10, 30
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
TOL = 1e-9


def vol_blocks(closes, n_blocks, window):
    n = len(closes)
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(float(np.std(np.log(closes[i - window : i]), ddof=1))
                            * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    block_size = n // n_blocks
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            bmed = float(
                np.median([v for v in bar_vols[b * block_size : (b + 1) * block_size]
                           if v > 0]) if block_size > window else 0.0
            )
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels):
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= MIN_SEGMENT_BARS:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= MIN_SEGMENT_BARS:
        runs.append((cur, run_start, len(labels)))
    return runs


def crossover(closes, fast, slow):
    n = len(closes)
    s = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(max(fast, slow) - 1, n):
        fm = np.mean(closes[i - fast + 1 : i + 1])
        sm = np.mean(closes[i - slow + 1 : i + 1])
        s[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
    return s


def adaptive_signals(closes, labels):
    """Fresh implementation of the regime-adaptive MA (calm 20/60, turbulent 10/30).

    Different code path from `research.backtest.regime_adaptive_ma_signals`: this
    one builds the labels on the fly (same algorithm as `volatility_blocks`) and
    applies the per-regime windows directly inside the signal loop. Same contract:
    past-only, same length as closes.
    """
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(n):
        lab = labels[i]
        if lab == "insufficient":
            continue
        fast, slow = (FAST_TURB, SLOW_TURB) if lab == "turbulent" else (FAST, SLOW)
        if i >= slow - 1:
            fm = np.mean(closes[i - fast + 1 : i + 1])
            sm = np.mean(closes[i - slow + 1 : i + 1])
            out[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
    return out


def filtered_signals(closes, fast, slow, labels):
    base = crossover(closes, fast, slow)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes))]
    for i in range(len(closes)):
        if i >= WINDOW and labels[i] == "turbulent":
            out[i] = base[i]
    return out


def segment_median(bars, signals, s, e, train, test, warm, overlap):
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = bt.BacktestConfig(warmup_periods=warm)
    res = bt.walk_forward(seg_bars, seg_signals, train_window=train,
                          test_window=test, warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def asset_medians(ticker, base_seed, signals_fn):
    np.random.seed(base_seed)
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels)
    signals = signals_fn(closes, labels)
    medians = []
    for lab, s, e in runs:
        m, nf = segment_median(bars, signals, s, e, TRAIN, TEST, WARM, OVERLAP)
        medians.append(round(m, 3))
    return medians, [lab for lab, _, _ in runs]


def main():
    np.random.seed(42)
    from research.data.preflight import load_manifest
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    published = {
        ("AMZN", "base"): ([+0.046, +0.001, -0.154], ["turbulent", "calm", "turbulent"]),
        ("AMZN", "adaptive"): ([-0.028, +0.001, +0.015],
                               ["turbulent", "calm", "turbulent"]),
        ("AMZN", "turb_only"): ([+0.046, +0.000, -0.154],
                                ["turbulent", "calm", "turbulent"]),
        ("JPM", "base"): ([+0.109, -0.006, +0.033], ["turbulent", "calm", "turbulent"]),
        ("JPM", "adaptive"): ([+0.069, -0.006, -0.001],
                              ["turbulent", "calm", "turbulent"]),
        ("JPM", "turb_only"): ([+0.109, +0.000, +0.033],
                               ["turbulent", "calm", "turbulent"]),
    }

    for (ticker, kind), (pub_med, pub_lbls) in published.items():
        np.random.seed(42)
        if kind == "base":
            comp = lambda c, lab: crossover(c, FAST, SLOW)
        elif kind == "adaptive":
            comp = lambda c, lab: adaptive_signals(c, lab)
        else:
            comp = lambda c, lab: filtered_signals(c, FAST, SLOW, lab)
        med, lbls = asset_medians(ticker, 42, comp)
        status = "MATCH" if med == pub_med and lbls == pub_lbls else "MISMATCH"
        print(f"  {ticker} {kind:8s}: recomputed {med} segments={lbls} "
              f"vs published {pub_med} segments={pub_lbls} -> {status}")
        assert med == pub_med and lbls == pub_lbls, f"{ticker} {kind} mismatch"

    print()
    print("Determinism: recomputing all six runs again and comparing.")
    results = {}
    for (ticker, kind), (pub_med, pub_lbls) in published.items():
        np.random.seed(42)
        if kind == "base":
            comp = lambda c, lab: crossover(c, FAST, SLOW)
        elif kind == "adaptive":
            comp = lambda c, lab: adaptive_signals(c, lab)
        else:
            comp = lambda c, lab: filtered_signals(c, FAST, SLOW, lab)
        results[(ticker, kind)] = asset_medians(ticker, 42, comp)

    for (ticker, kind), (med, lbls) in results.items():
        np.random.seed(42)
        if kind == "base":
            comp = lambda c, lab: crossover(c, FAST, SLOW)
        elif kind == "adaptive":
            comp = lambda c, lab: adaptive_signals(c, lab)
        else:
            comp = lambda c, lab: filtered_signals(c, FAST, SLOW, lab)
        med2, lbls2 = asset_medians(ticker, 42, comp)
        assert med == med2 and lbls == lbls2, f"{ticker} {kind} non-deterministic"
        print(f"  {ticker} {kind}: deterministic (identical across reruns)")

    print("\nAll independent recomputations MATCH the adaptive check figures.")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Regime-adaptive MA crossover verification on AMZN/JPM: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
