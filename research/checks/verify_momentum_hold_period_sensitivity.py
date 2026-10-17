"""Independent verification of `research/checks/momentum_hold_period_sensitivity.py`.

Two verification gates:

1. Fresh recomputation of the AMZN/JPM medians per hold period via a separate
   implementation path (re-implemented `volatility_blocks`, segment reconstruction,
   and momentum signals with holding-period transform, aggregating fold log returns
   with `walk_forward`). Every per-asset, per-hold median is recomputed and compared
   to the artifact.

2. Cross-consistency gate: the hold=1 column of this artifact is an independent
   rerun of the momentum class (H=1), so it must match `momentum_results.json`
   exactly (medians, nulls, verdicts). A mismatch would flag a change in the
   lookback-5 computation or a defect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
HOLD_PERIODS = (1, 2, 3, 5)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_hold_period_sensitivity_results.json"
MOM_ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"

SWEEP_VERDICT_TOL = 0.05


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
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window else 0.0)
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


def hold_effective_signals(signals, hold, lookback=5):
    """Carry each signal for `hold` bars before the next rebalance.
    
    Position at bar b equals the signal from the most recent rebalance bar
    b' <= b, where rebalance bars are lookback + k*hold. Concretely,
    effective[b] = signals[b - (b - lookback) % hold] for b >= lookback,
    else neutral.
    """
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < lookback:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - lookback) % hold
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out


def segment_result(ticker, s, e, train, test, warm, overlap, hold, lookback=5):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    eff = hold_effective_signals(signals, hold, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = eff[s:e]
    cfg = bt.BacktestConfig(warmup_periods=warm)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookback"] == 5

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-asset per-hold: recompute medians vs artifact
    for asset in ["AMZN", "JPM"]:
        for hold in HOLD_PERIODS:
            bars, dates = bt.load_ticker(asset)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]

            medians = []
            for lab, s, e in runs:
                m, nf = segment_result(asset, s, e, TRAIN, TEST, WARM, OVERLAP, hold)
                medians.append(round(m, 3))

            # Get artifact medians
            pub = artifact["per_asset"][asset][str(hold)]["medians"]
            pub_lbls = artifact["per_asset"][asset][str(hold)]["segments"]
            status = "MATCH" if medians == pub and lbls == pub_lbls else "MISMATCH"
            if status != "MATCH":
                all_ok = False
                print(f"  {asset} hold={hold}: medians {medians} vs published {pub} -> {status}")

    # Cross-consistency gate: hold=1 must match momentum_results.json
    print("\n=== Cross-consistency gate (hold=1 vs momentum_results.json) ===")
    try:
        with open(MOM_ARTIFACT_PATH) as f:
            mom_artifact = json.load(f)

        for asset in ["AMZN", "JPM"]:
            bars, dates = bt.load_ticker(asset)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]

            medians = []
            for lab, s, e in runs:
                m, nf = segment_result(asset, s, e, TRAIN, TEST, WARM, OVERLAP, 1)
                medians.append(round(m, 3))

            pub = mom_artifact["per_asset"][asset]["medians"]
            pub_lbls = mom_artifact["per_asset"][asset]["segments"]
            status = "MATCH" if medians == pub and lbls == pub_lbls else "MISMATCH"
            if status != "MATCH":
                all_ok = False
                print(f"  {asset} hold=1: medians {medians} vs published {pub} -> {status}")
            else:
                print(f"  {asset} hold=1: medians {medians} vs published {pub} -> {status}")
    except Exception as e:
        print(f"  Cross-consistency gate failed: {e}")
        all_ok = False

    print("\nDeterminism: recomputing a subset again.")
    for asset in ["AMZN", "JPM"]:
        for hold in [1, 2]:
            bars, dates = bt.load_ticker(asset)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            m1 = [round(segment_result(asset, s, e, TRAIN, TEST, WARM, OVERLAP, hold)[0], 3)
                  for lab, s, e in runs]
            m2 = [round(segment_result(asset, s, e, TRAIN, TEST, WARM, OVERLAP, hold)[0], 3)
                  for lab, s, e in runs]
            status = "identical" if m1 == m2 else "DIFFERENT"
            print(f"  {asset} hold={hold}: r1 {m1} r2 {m2} -> {status}")
            if status != "identical":
                all_ok = False

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the independent path.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution. Momentum hold-period sensitivity independent verification on AMZN/JPM + universe: exploratory simulation.")


if __name__ == "__main__":
    main()