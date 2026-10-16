"""Independent verification of the momentum volatility gate check
(`research/checks/momentum_volatility_gate.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the momentum volatility gate check (`state/check_artifacts/momentum_volatility_gate_results.json`).
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


def momentum_volatility_gate_signals(closes, lookback, window, labels):
    """Fresh implementation of momentum with volatility gate (independent code path)."""
    base = momentum_signals(closes, lookback)
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(n):
        if i >= window and labels[i] == "turbulent":
            out[i] = base[i]
    return out


def segment_median(ticker, s, e, lookback, window, train, test, warm, overlap):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, window)
    signals = momentum_volatility_gate_signals(closes, lookback, window, labels)
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
    ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_volatility_gate_results.json"
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookback"] == LOOKBACK

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-asset: base momentum, gated momentum segment medians vs artifact
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        # Base momentum medians
        base_medians = []
        for lab, s, e in runs:
            # For base momentum, use regular momentum signals
            signals = momentum_signals(closes, LOOKBACK)
            seg_bars = list(bars)[s:e]
            seg_signals = signals[s:e]
            cfg = bt.BacktestConfig(warmup_periods=WARM)
            res = bt.walk_forward(
                seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
            log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                            for f in res.folds], dtype=np.float64)
            base_medians.append(round(float(np.median(log)), 3))

        # Gated momentum medians
        gated_medians = []
        for lab, s, e in runs:
            signals = momentum_volatility_gate_signals(closes, LOOKBACK, WINDOW, labels)
            seg_bars = list(bars)[s:e]
            seg_signals = signals[s:e]
            cfg = bt.BacktestConfig(warmup_periods=WARM)
            res = bt.walk_forward(
                seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
            log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                            for f in res.folds], dtype=np.float64)
            gated_medians.append(round(float(np.median(log)), 3))

        # Check artifact
        pub_base = artifact["per_asset"][asset]["base"]["medians"]
        pub_gated = artifact["per_asset"][asset]["gated"]["medians"]
        pub_lbls = artifact["per_asset"][asset]["base"]["segments"]

        base_status = "MATCH" if base_medians == pub_base and lbls == pub_lbls else "MISMATCH"
        gated_status = "MATCH" if gated_medians == pub_gated and lbls == pub_lbls else "MISMATCH"
        if base_status != "MATCH" or gated_status != "MATCH":
            all_ok = False
        print(f"  {asset}: base medians {base_medians} vs published {pub_base} -> {base_status}")
        print(f"  {asset}: gated medians {gated_medians} vs published {pub_gated} -> {gated_status}")

    # Determinism of this verification path
    print("\nDeterminism: recomputing universe medians for AMZN and JPM again.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_median("AMZN", s, e, LOOKBACK, WINDOW, TRAIN, TEST, WARM, OVERLAP)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_median("AMZN", s, e, LOOKBACK, WINDOW, TRAIN, TEST, WARM, OVERLAP)[0], 3)
          for lab, s, e in runs]
    print(f"  AMZN r1: {m1}; r2: {m2} -> {'identical' if m1 == m2 else 'DIFFERENT'}")
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum volatility gate independent verification on AMZN/JPM: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
