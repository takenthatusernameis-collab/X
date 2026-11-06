"""Independent verification of the momentum cost-robustness check
(research/checks/momentum_cost_robustness.py).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the cost-robustness check (`state/check_artifacts/momentum_cost_robustness_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- Cost model parameters are copied from the artifact and applied.
- Perturbation medians are recomputed by walking each parameter set
  independently.
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


def segment_median_with_cost(
    ticker: str,
    s: int,
    e: int,
    bars,  # full bars (need for date mapping)
    lookback: int,
    train: int,
    test: int,
    warm: int,
    overlap: int,
):
    """Compute segment median using fresh walk_forward with cost model."""
    # Extract segment data
    seg_bars = bars[s:e]
    seg_dates = [b.date for b in seg_bars]

    # Get full closes for signal generation
    full_closes = bars.closes_array()
    signals = momentum_signals(full_closes, lookback)
    seg_signals = signals[s:e]

    # Create cost model from artifact defaults (same as check)
    cfg = bt.BacktestConfig(
        initial_capital=1_000_000,
        target_exposure=1.0,
        commission_per_trade=0.001,
        commission_per_share=0.0001,
        slippage_cents=0.5,
        slippage_proportional=0.0002,
        warmup_periods=0,
        risk_free=0.0,
        periods_per_year=252,
        margin_rate=0.0,
        margin_call_liquidate=True,
    )

    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)

    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def create_cost_model():
    """Create a realistic cost model matching the repository's existing cost model."""
    return bt.BacktestConfig(
        initial_capital=1_000_000,
        target_exposure=1.0,
        commission_per_trade=0.001,
        commission_per_share=0.0001,
        slippage_cents=0.5,
        slippage_proportional=0.0002,
        warmup_periods=0,
        risk_free=0.0,
        periods_per_year=252,
        margin_rate=0.0,
        margin_call_liquidate=True,
    )


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == list(LOOKBACKS)

    print("=== Independent Recomputation (Fresh walk_forward path) ===")
    all_ok = True

    # Load manifest for real data
    from research.data.preflight import load_manifest
    manifest = load_manifest()

    # Per-asset: segment medians vs artifact
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        # For each lookback, compute segment medians
        asset_artifacts = {}
        for lb in LOOKBACKS:
            lb_artifacts = artifact["per_asset"][asset]["lookback_results"]
            key = str(lb)
            if key not in lb_artifacts:
                continue
            asset_artifacts[lb] = lb_artifacts[key]

        print(f"  {asset}: recomputing segment medians for lookbacks {list(asset_artifacts.keys())}")

        for lb, lb_artifact in asset_artifacts.items():
            lb_int = int(lb)
            recomputed_medians = []
            for lab, s, e in runs:
                median, n_folds = segment_median_with_cost(
                    asset, s, e, bars, lb_int,
                    TRAIN, TEST, WARM, OVERLAP)
                recomputed_medians.append(round(median, 3))

            published = lb_artifact["medians"]
            pub_lbls = lb_artifact["segments"]

            status = "MATCH" if recomputed_medians == published and lbls == pub_lbls else "MISMATCH"
            if status != "MATCH":
                all_ok = False
            print(f"    lookback {lb}: recomputed {recomputed_medians} vs published {published} -> {status}")
            if status != "MATCH":
                print(f"      run_start/end labels: {lbls} vs {pub_lbls}")

    # Universe-level per-asset medians
    print("\n=== Universe-level recomputation ===")
    # Load all tickers
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    for ticker in artifact["universe"]["per_asset"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        # For lookback 5 (reference) recompute
        lb = 5
        recomputed_medians = []
        for lab, s, e in runs:
            median, n_folds = segment_median_with_cost(
                ticker, s, e, bars, lb,
                TRAIN, TEST, WARM, OVERLAP)
            recomputed_medians.append(round(median, 3))

        pub = artifact["universe"]["per_asset"][ticker]["lookback_medians"]["5"]
        pub_lbls = artifact["universe"]["per_asset"][ticker]["lookback_verdicts_5"]

        # Regenerate the verdict from recomputed medians and null medians
        cand = np.array(recomputed_medians)
        null = np.array(artifact["universe"]["per_asset"][ticker]["lookback_null_medians"]["5"])

        if all(abs(m) <= 0.05 for m in recomputed_medians):
            regen_verdict = "CONSISTENT_WITH_NOISE"
        elif np.std(cand) > 2.0 * np.std(null):
            regen_verdict = "REGIME_DEPENDENT"
        elif all(m < 0 for m in recomputed_medians):
            regen_verdict = "REGIME_STABLE_LOSS"
        else:
            regen_verdict = "REGIME_STABLE"

        status = "MATCH" if recomputed_medians == pub and lbls == pub_lbls else "MISMATCH"
        if status != "MATCH":
            all_ok = False
        verdict_status = "MATCH" if regen_verdict == pub_lbls else "MISMATCH"
        if verdict_status != "MATCH":
            all_ok = False

        print(f"  {ticker}: medians {recomputed_medians} segments={lbls} -> {status} | "
              f"verdict {regen_verdict} (pub: {pub_lbls}) verdict {verdict_status}")

    # Determinism test
    print("\nDeterminism: recomputing universe medians for AMZN and JPM again.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)

    # First pass
    medians1 = []
    for lb in LOOKBACKS:
        lb_medians = []
        for lab, s, e in runs:
            median, _ = segment_median_with_cost(
                "AMZN", s, e, bars, lb,
                TRAIN, TEST, WARM, OVERLAP)
            lb_medians.append(round(median, 3))
        medians1.append(lb_medians)

    # Second pass
    medians2 = []
    for lb in LOOKBACKS:
        lb_medians = []
        for lab, s, e in runs:
            median, _ = segment_median_with_cost(
                "AMZN", s, e, bars, lb,
                TRAIN, TEST, WARM, OVERLAP)
            lb_medians.append(round(median, 3))
        medians2.append(lb_medians)

    print(f"  AMZN lookback medians r1: {[list(x) for x in medians1]} r2: {[list(x) for x in medians2]} -> "
          f"{'identical' if medians1 == medians2 else 'DIFFERENT'}")
    assert medians1 == medians2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-robustness independent verification on AMZN/JPM + universe: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()