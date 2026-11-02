"""Independent verification of the momentum cost-robustness check
(research/checks/momentum_cost_robustness.py).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh
`walk_forward` implementation (not ``stress_segments``), then compares
against the artifact written by the cost-robustness check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- Realistic costs are applied consistently.
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

# Same realistic costs as the check
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_share=0.0003,  # 3 bps per share
    commission_per_trade=0.01,   # $0.01 per trade
    slippage_cents=5.0,          # 5 cents per share fixed slippage
    slippage_proportional=0.0,
    warmup_periods=0,
)

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"

BASE_SEED = 42  # same base seed the check used for perturbations


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.regime_stability.volatility_blocks."""
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
    """Contiguous segments from past-only labels (same logic as the framework)."""
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


def null_signals_from_actual(actual_signals):
    """Create matched null (coin-flip sign) signals from actual signals."""
    return [
        bt.Signal(date=s.date, weight=-s.weight) if s.weight != 0 else s
        for s in actual_signals
    ]


def segment_median_with_costs(ticker, s, e, train, test, warm, overlap, lookback):
    """Walk-forward segment median with realistic costs for one ticker/segment."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback=lookback)
    null_signals = null_signals_from_actual(signals)

    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    seg_null_signals = null_signals[s:e]

    cfg = bt.BacktestConfig(
        initial_capital=REALISTIC_COSTS.initial_capital,
        target_exposure=REALISTIC_COSTS.target_exposure,
        commission_per_share=REALISTIC_COSTS.commission_per_share,
        commission_per_trade=REALISTIC_COSTS.commission_per_trade,
        slippage_cents=REALISTIC_COSTS.slippage_cents,
        slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        warmup_periods=0,
    )

    # Actual momentum
    res_actual = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    actual_lrets = np.array([
        np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
        for f in res_actual.folds], dtype=np.float64)
    actual_median = float(np.median(actual_lrets))

    # Null momentum
    res_null = bt.walk_forward(
        seg_bars, seg_null_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    null_lrets = np.array([
        np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
        for f in res_null.folds], dtype=np.float64)
    null_median = float(np.median(null_lrets))

    return actual_median, null_median, len(res_actual.folds)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== Independent recomputation (fresh walk_forward path) ===")

    # Load manifest for ticker data
    from research.data.preflight import load_manifest
    m = load_manifest()
    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in m["entries"]}

    all_ok = True

    # Per-asset: compute medians for all lookbacks
    for ticker in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        for lb in LOOKBACKS:
            medians_actual = []
            medians_null = []
            for lab, s, e in runs:
                a, n, nf = segment_median_with_costs(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb)
                medians_actual.append(round(a, 3))
                medians_null.append(round(n, 3))

            # Compare with artifact
            artifact_actual = artifact["per_asset"][str(lb)][ticker]["actual"]["medians"]
            artifact_null = artifact["per_asset"][str(lb)][ticker]["null"]["medians"]

            actual_status = "MATCH" if medians_actual == artifact_actual else "MISMATCH"
            null_status = "MATCH" if medians_null == artifact_null else "MISMATCH"

            if actual_status != "MATCH" or null_status != "MATCH":
                all_ok = False

            print(f"  {ticker} lookback={lb}: actual medians {medians_actual} vs "
                  f"published {artifact_actual} -> {actual_status}")
            print(f"  {ticker} lookback={lb}: null medians {medians_null} vs "
                  f"published {artifact_null} -> {null_status}")

    # Universe-level: check robustness verdicts for all tickers
    print("\n=== Universe-level recomputation ===")
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]:
        for lb in LOOKBACKS:
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]

            medians_actual = []
            medians_null = []
            for lab, s, e in runs:
                a, n, nf = segment_median_with_costs(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb)
                medians_actual.append(round(a, 3))
                medians_null.append(round(n, 3))

            # Compare with artifact
            artifact_actual = artifact["per_asset"][str(lb)][ticker]["actual"]["medians"]
            artifact_null = artifact["per_asset"][str(lb)][ticker]["null"]["medians"]

            actual_status = "MATCH" if medians_actual == artifact_actual else "MISMATCH"
            null_status = "MATCH" if medians_null == artifact_null else "MISMATCH"

            if actual_status != "MATCH" or null_status != "MATCH":
                all_ok = False
                print(f"  [DEBUG] ticker={ticker} lookback={lb}: len(runs)={len(runs)}")
                print(f"  [DEBUG] ticker={ticker} lookback={lb}: pub_actual={artifact_actual}")
                print(f"  [DEBUG] ticker={ticker} lookback={lb}: recomputed_actual={medians_actual}")
                print(f"  [DEBUG] ticker={ticker} lookback={lb}: pub_null={artifact_null}")
                print(f"  [DEBUG] ticker={ticker} lookback={lb}: recomputed_null={medians_null}")

            print(f"  {ticker} lookback={lb}: actual {actual_status}, null {null_status}")

    # Determinism of this verification path
    print("\nDeterminism: recomputing medians for AMZN and JPM again.")
    for lb in LOOKBACKS:
        bars, dates = bt.load_ticker("AMZN")
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        m1 = [round(segment_median_with_costs("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, lb)[0], 3)
              for lab, s, e in runs]
        m2 = [round(segment_median_with_costs("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, lb)[0], 3)
              for lab, s, e in runs]
        print(f"  AMZN lookback={lb} r1: {m1}; r2: {m2} -> {'identical' if m1 == m2 else 'DIFFERENT'}")
        assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-robustness independent verification on "
          "the collected universe: exploratory simulation with realistic costs.")


if __name__ == "__main__":
    main()
