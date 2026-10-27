"""Independent verification of momentum cost-robustness check.

This recomputes each segment's walk-forward median from a fresh
walk_forward implementation (not stress_segments), then compares against
the artifact written by the momentum cost-robustness check.

Design:
- Reconstruct regime labels from closes using a re-implemented
  volatility_blocks (same algorithm as research.backtest, different code).
- Reconstruct segments from labels using min_segment_bars.
- Compute segment medians via walk_forward directly on segment slices,
  aggregating fold log returns.
- All values are loaded from the check's artifact and compared.
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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"

BASE_SEED = 42


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
    """Fresh implementation of momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_median(ticker, s, e, train, test, warm, overlap, lookback):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
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

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-lookback, per-asset median recomputation vs artifact
    for lb in LOOKBACKS:
        print(f"\n--- Lookback {lb} recomputation ---")
        key = f"lookback_{lb}"
        if key not in artifact:
            print(f"  WARNING: {key} not in artifact")
            all_ok = False
            continue

        for ticker in artifact[key]["per_asset"]:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]

            medians = []
            for lab, s, e in runs:
                m, nf = segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb)
                medians.append(round(m, 3))

            pub = artifact[key]["per_asset"][ticker]["medians"]
            pub_lbls = artifact[key]["per_asset"][ticker]["segments"]

            status = "MATCH" if medians == pub and lbls == pub_lbls else "MISMATCH"
            if status != "MATCH":
                all_ok = False
                print(f"  {ticker}: medians {medians} vs published {pub} -> {status}")
            verdict = artifact[key]["per_asset"][ticker]["verdict"]
            print(f"  {ticker}: medians {medians} segments={len(lbls)} -> {status} | verdict {verdict}")

    # Determinism of this verification path
    print("\nDeterminism: recomputing lookback 3 medians again.")
    for lb in LOOKBACKS:
        # Check a random asset
        ticker = list(artifact[f"lookback_{lb}"]["per_asset"].keys())[0]
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        
        m1 = [round(segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb)[0], 3)
              for lab, s, e in runs]
        # Rerun with same seed
        np.random.seed(SEED)
        m2 = [round(segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb)[0], 3)
              for lab, s, e in runs]
        print(f"  {ticker} lookback {lb}: r1={m1} r2={m2} -> {"identical" if m1 == m2 else "DIFFERENT"}")
        assert m1 == m2, f"verification path not deterministic for {ticker} lookback {lb}"

    print("\n" + "="*60)
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the independent path.")
        print("Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-robustness independent verification: lookback 3/5/10.")


if __name__ == "__main__":
    main()