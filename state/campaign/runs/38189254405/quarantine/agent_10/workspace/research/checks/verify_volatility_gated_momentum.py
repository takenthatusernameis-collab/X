"""Independent verification of the volatility-gated momentum check
(`research/checks/volatility_gated_momentum.py`).

This is a separate implementation path: it recomputes each segment's walk-forward
median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the volatility-gated momentum check (`state/check_artifacts/volatility_gated_momentum_results.json`).
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
V_THRESHOLD = 0.20  # predeclared volatility threshold (annualized realized vol)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "volatility_gated_momentum_results.json"


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
    """Pure momentum signal implementation for independent verification."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def volatility_gate_momentum_signals(closes, lookback, v_threshold, vol_window):
    """Independent implementation of volatility-gated momentum signals."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    sqrt252 = np.sqrt(252.0)
    
    # First compute base momentum signals
    momentum_sig = momentum_signals(closes, lookback)
    
    # Apply volatility gate
    for i in range(vol_window - 1, n):
        # Calculate realized volatility over trailing vol_window bars
        log_returns = np.log(closes[i - vol_window + 1 : i + 1])
        realized_annual = float(np.std(log_returns, ddof=1)) * sqrt252
        
        if realized_annual < v_threshold:
            # Volatility below threshold: use momentum signal
            out[i] = momentum_sig[i]
        # else: weight remains 0 (neutral)
    
    # Handle bars before vol_window (insufficient data)
    for i in range(vol_window - 1):
        out[i] = momentum_sig[i]
    
    return out


def segment_median(ticker, s, e, lookback, v_threshold, vol_window, train, test, warm, overlap):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = volatility_gate_momentum_signals(closes, lookback, v_threshold, vol_window)
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
    assert artifact["v_threshold"] == V_THRESHOLD
    assert artifact["vol_window"] == WINDOW

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-asset verification
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        medians = []
        for lab, s, e in runs:
            m, nf = segment_median(asset, s, e, LOOKBACK, V_THRESHOLD, WINDOW, TRAIN, TEST, WARM, OVERLAP)
            medians.append(round(m, 3))

        pub_med = artifact["per_asset"][asset]["vol_gated"]["medians"]
        pub_lbls = artifact["per_asset"][asset]["vol_gated"]["segments"]

        status = "MATCH" if medians == pub_med and lbls == pub_lbls else "MISMATCH"
        if status != "MATCH":
            all_ok = False
        print(f"  {asset}: recomputed {medians} segments={lbls} "
              f"vs published {pub_med} segments={pub_lbls} -> {status}")

    # Universe-level recomputation
    print("\n=== Universe-level recomputation ===")
    tickers = {}
    from research.data.preflight import load_manifest
    m = load_manifest()
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    for ticker in artifact["universe"]["per_asset"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        medians = [round(segment_median(ticker, s, e, LOOKBACK, V_THRESHOLD, WINDOW, TRAIN, TEST, WARM, OVERLAP)[0], 3)
                   for lab, s, e in runs]
        pub = artifact["universe"]["per_asset"][ticker]["medians"]
        pub_lbls = artifact["universe"]["per_asset"][ticker]["segments"]

        status = "MATCH" if medians == pub and lbls == pub_lbls else "MISMATCH"
        if status != "MATCH":
            all_ok = False
            print(f"  [DEBUG] ticker={ticker}: len(runs)={len(runs)} lbls={lbls}")
            print(f"  [DEBUG] ticker={ticker}: pub_lbls={pub_lbls}")
            print(f"  [DEBUG] ticker={ticker}: len(medians)={len(medians)} medians={medians}")
            print(f"  [DEBUG] ticker={ticker}: pub_medians={pub}")

        verdict = "CONSISTENT_WITH_NOISE" if all(abs(m) <= 0.05 for m in medians) else \
            "REGIME_DEPENDENT" if np.std(medians) > 2.0 * np.std(artifact["universe"]["per_asset"][ticker]["null_medians"]) else \
            "REGIME_STABLE_LOSS" if all(m < 0 for m in medians) else "REGIME_STABLE"
        verdict_status = "MATCH" if verdict == artifact["universe"]["per_asset"][ticker]["verdict"] else "MISMATCH"
        if verdict_status != "MATCH":
            all_ok = False
        print(f"  {ticker}: medians {medians} segments={pub_lbls} -> {status} | verdict {verdict} (pub: {artifact['universe']['per_asset'][ticker]['verdict']})")

    # Determinism of this verification path
    print("\nDeterminism: recomputing universe medians for AMZN and JPM again.")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        m1 = [round(segment_median(asset, s, e, LOOKBACK, V_THRESHOLD, WINDOW, TRAIN, TEST, WARM, OVERLAP)[0], 3)
              for lab, s, e in runs]
        m2 = [round(segment_median(asset, s, e, LOOKBACK, V_THRESHOLD, WINDOW, TRAIN, TEST, WARM, OVERLAP)[0], 3)
              for lab, s, e in runs]
        print(f"  {asset} r1: {m1}; r2: {m2} -> {'identical' if m1 == m2 else 'DIFFERENT'}")
        assert m1 == m2, f"verification path not deterministic for {asset}"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Volatility-gated momentum independent verification on AMZN/JPM + universe: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
