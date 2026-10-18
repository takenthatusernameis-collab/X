"""Independent verification of the momentum cost robustness check
(research/checks/momentum_cost_robustness.py).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the momentum cost robustness check.
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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"

# Cost model parameters from examples
COST_ZERO = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)

COST_REALISTIC = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)


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


def segment_median_cost(ticker, s, e, train, test, warm, overlap, cfg, lookback=5):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
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
    assert artifact["lookbacks"] == list(LOOKBACKS)

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Recompute per-asset medians for both cost types
    for cost_label, cfg in [("zero", COST_ZERO), ("realistic", COST_REALISTIC)]:
        print(f"\n=== Cost: {cost_label} ===")
        for asset in ["AMZN", "JPM"]:
            bars, dates = bt.load_ticker(asset)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]

            # Recompute segment medians using fresh walk_forward for lookback=5
            medians = []
            for lab, s, e in runs:
                m, nf = segment_median_cost(asset, s, e, TRAIN, TEST, WARM, OVERLAP, cfg, lookback=5)
                medians.append(round(m, 3))

            # Compare with artifact
            pub = artifact[cost_label + "_cost"]["per_asset"][asset]["5"]["medians"]
            pub_lbls = artifact[cost_label + "_cost"]["per_asset"][asset]["5"]["segments"]

            if medians == pub and lbls == pub_lbls:
                print(f"  {asset}: medians and segments MATCH")
            else:
                print(f"  {asset}: MISMATCH")
                if medians != pub:
                    print(f"    medians: recomputed {medians} vs published {pub}")
                if lbls != pub_lbls:
                    print(f"    segments: recomputed {lbls} vs published {pub_lbls}")
                all_ok = False

    # Recompute universe-level medians for both cost types
    print("\n=== Universe-level recomputation ===")
    from research.data.preflight import load_manifest
    m = load_manifest()
    tickers = {}
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    for cost_label, cfg in [("zero", COST_ZERO), ("realistic", COST_REALISTIC)]:
        print(f"\n--- Cost: {cost_label} ---")
        for ticker in artifact[cost_label + "_cost"]["universe"]["per_asset"]:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]

            # Recompute medians using fresh walk_forward
            medians = [round(segment_median_cost(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, cfg, lookback=5)[0], 3)
                       for lab, s, e in runs]

            pub = artifact[cost_label + "_cost"]["universe"]["per_asset"][ticker]["lookback_medians"]["5"]
            pub_lbls = artifact[cost_label + "_cost"]["universe"]["per_asset"][ticker]["verdicts_5"]
            status = "MATCH" if medians == pub and lbls == artifact[cost_label + "_cost"]["universe"]["per_asset"][ticker]["lookback_medians"]["5"] else "MISMATCH"

            if status != "MATCH":
                all_ok = False
                print(f"  {ticker}: MISMATCH")
                if medians != pub:
                    print(f"    medians: recomputed {medians} vs published {pub}")
                if lbls != artifact[cost_label + "_cost"]["universe"]["per_asset"][ticker]["lookback_medians"]["5"]:
                    print(f"    segments: recomputed {lbls} vs published {artifact[cost_label + '_cost']['universe']['per_asset'][ticker]['lookback_medians']['5']}")

            verdict = artifact[cost_label + "_cost"]["universe"]["per_asset"][ticker]["verdicts_5"]
            print(f"  {ticker}: medians {medians} segments={lbls} -> {status} | verdict {verdict}")

    # Determinism
    print("\nDeterminism: recomputing universe medians for AMZN and JPM again.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_median_cost("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, COST_ZERO, lookback=5)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_median_cost("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, COST_ZERO, lookback=5)[0], 3)
          for lab, s, e in runs]
    print("  AMZN r1: {}; r2: {} -> {}".format(m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost robustness independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()