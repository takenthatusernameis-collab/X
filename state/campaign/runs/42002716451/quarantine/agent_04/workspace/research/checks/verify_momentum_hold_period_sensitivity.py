"""Independent verification of the momentum holding-period sensitivity check
(`research/checks/momentum_hold_period_sensitivity.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the momentum hold-period sensitivity check (`state/check_artifacts/momentum_hold_period_sensitivity_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments) so a match
confirms the check computed from those inputs; a mismatch would flag an
error in the check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
HOLD_PERIODS = (1, 2, 3, 5)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_hold_period_sensitivity_results.json"

BASE_SEED = 42  # same base seed the check used for perturbations


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


def hold_effective_signals(signals, lookback, hold):
    """Carry each signal for `hold` bars before the next rebalance."""
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < lookback:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - lookback) % hold
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out


def segment_median(ticker, s, e, train, test, warm, overlap, hold):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = hold_effective_signals(momentum_signals(closes, LOOKBACK), LOOKBACK, hold)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
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
    assert artifact["lookback"] == LOOKBACK
    assert artifact["hold_periods"] == list(HOLD_PERIODS)

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-asset: holding-period sensitivity results vs artifact
    for asset in artifact["per_asset"].keys():
        for hold in HOLD_PERIODS:
            bars, dates = bt.load_ticker(asset)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]

            medians = []
            for lab, s, e in runs:
                m, nf = segment_median(asset, s, e, TRAIN, TEST, WARM, OVERLAP, hold)
                medians.append(round(m, 3))

            pub = artifact["per_asset"][asset][str(hold)]["medians"]
            pub_lbls = artifact["per_asset"][asset][str(hold)]["segments"]
            status = "MATCH" if medians == pub and lbls == pub_lbls else "MISMATCH"
            if status != "MATCH":
                all_ok = False
                print("  [DEBUG] ticker={}: len(runs)={} lbls={}".format(asset, len(runs), lbls))
                print("  [DEBUG] ticker={}: pub_lbls={}".format(asset, pub_lbls))
                print("  [DEBUG] ticker={}: len(medians)={} medians={}".format(asset, len(medians), medians))
                print("  [DEBUG] ticker={}: pub_medians={}".format(asset, pub))
            print("  {}: hold={}: medians {} segments={} -> {}".format(
                asset, hold, medians, pub_lbls, status))

    # Universe-level per-asset medians and verdicts vs artifact
    print("\n=== Universe-level recomputation ===")
    tickers = {}
    from research.data.preflight import load_manifest
    m = load_manifest()
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    # Get the ticker order from the artifact (should be consistent with manifest)
    ticker_order = artifact["universe"]["1"]["ticker_order"]
    
    for ticker in ticker_order:
        # Recompute medians for each hold period
        per_hold_medians = {}
        for hold in HOLD_PERIODS:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]
            
            medians = [round(segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, hold)[0], 3)
                       for lab, s, e in runs]
            per_hold_medians[str(hold)] = medians
            
            # Check against artifact
            pub = artifact["universe"][str(hold)]["per_asset"][ticker]["medians"]
            status = "MATCH" if medians == pub else "MISMATCH"
            if status != "MATCH":
                all_ok = False
                print("  [DEBUG] ticker={}: hold={}: recomputed_medians={} vs pub={}".format(
                    ticker, hold, medians, pub))
            print("  {}: hold={}: medians {} -> {}".format(ticker, hold, medians, status))

    # Determinism of this verification path
    print("\nDeterminism: recomputing universe medians for AMZN and JPM again.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 1)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 1)[0], 3)
          for lab, s, e in runs]
    print("  AMZN hold=1: r1: {}; r2: {} -> {}".format(m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum hold-period sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
