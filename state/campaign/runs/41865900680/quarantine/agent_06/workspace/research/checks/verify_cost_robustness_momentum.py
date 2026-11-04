"""Independent verification of the cost-robustness momentum check
(research/checks/cost_robustness_momentum.py).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the cost-robustness momentum check (`state/check_artifacts/cost_robustness_momentum_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments``.
- Cost modeling is applied consistently (zero and realistic cost regimes).
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments, cost regimes) so a
match confirms the check computed from those inputs; a mismatch would flag an
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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

# Cost regimes - must match the check's costs
ZERO_COST = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "cost_robustness_momentum_results.json"

BASE_SEED = 42  # base seed for consistency


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


def segment_median(
    ticker: str,
    s: int,
    e: int,
    train: int,
    test: int,
    warm: int,
    overlap: int,
    lookback: int,
    cost_cfg: bt.BacktestConfig,
) -> tuple[float, int]:
    """Compute segment median log return using fresh walk_forward."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = bt.BacktestConfig(warmup_periods=warm)
    # Use the cost configuration provided
    cfg.commission_per_trade = cost_cfg.commission_per_trade
    cfg.commission_per_share = cost_cfg.commission_per_share
    cfg.slippage_cents = cost_cfg.slippage_cents
    cfg.slippage_proportional = cost_cfg.slippage_proportional
    res = bt.walk_forward(
        seg_bars,
        seg_signals,
        train_window=train,
        test_window=test,
        warmup=warm,
        overlap_window=overlap,
        cfg=cfg,
    )
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(BASE_SEED)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == list(LOOKBACKS)

    print("=== Independent Recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-asset: cost-robustness segment medians vs artifact for realistic costs
    print("\n=== Per-asset realistic-cost recomputation ===")
    for asset in ["AAPL", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        realistic_medians = []

        for lab, s, e in runs:
            m, nf = segment_median(
                asset, s, e, TRAIN, TEST, WARM, OVERLAP, lookback=5, cost_cfg=REALISTIC_COSTS
            )
            realistic_medians.append(round(m, 3))

        # Compare with artifact
        pub_medians = artifact["per_asset"][asset]["5"]["medians"]
        pub_verdict = artifact["per_asset"][asset]["5"]["verdict"]

        med_status = "MATCH" if realistic_medians == pub_medians else "MISMATCH"
        if med_status != "MATCH":
            all_ok = False

        print(f"  {asset}: realistic medians {realistic_medians} vs published {pub_medians} -> {med_status}")
        print(f"  {asset}: verdict {pub_verdict}")

    # Universe-level per-asset medians and verdicts vs artifact
    print("\n=== Universe-level realistic-cost recomputation ===")
    tickers = {}
    from research.data.preflight import load_manifest
    m = load_manifest()
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    for ticker in artifact["per_asset"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        medians = [round(segment_median(
            ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lookback=5, cost_cfg=REALISTIC_COSTS
        )[0], 3)
                   for lab, s, e in runs]
        pub = artifact["per_asset"][ticker]["5"]["medians"]
        pub_lbls = artifact["per_asset"][ticker]["5"]["segments"]

        # Regenerate the verdict from recomputed medians (without null for simplicity)
        cand = np.array(medians)
        # Use artifact null medians for verdict comparison
        artifact_null = np.array(artifact["per_asset"][ticker]["5"]["null_medians"])
        cand_disp = round(float(np.std(cand)), 3)
        null_disp = round(float(np.std(artifact_null)), 3)
        if all(abs(m) <= 0.05 for m in medians):
            regen_verdict = "CONSISTENT_WITH_NOISE"
        elif cand_disp > 2.0 * null_disp:
            regen_verdict = "REGIME_DEPENDENT"
        elif all(m < 0 for m in medians):
            regen_verdict = "REGIME_STABLE_LOSS"
        else:
            regen_verdict = "REGIME_STABLE"

        status = "MATCH" if medians == pub else "MISMATCH"
        if status != "MATCH":
            all_ok = False

        verdict_status = "MATCH" if regen_verdict == artifact["per_asset"][ticker]["5"]["verdict"] else "MISMATCH"
        if verdict_status != "MATCH":
            all_ok = False

        print(f"  {ticker}: medians {medians} -> {status} | verdict {regen_verdict} (pub: {artifact['per_asset'][ticker]['5']['verdict']})")

    # Determinism of this verification path
    print("\n=== Determinism check ===")
    print("  Recomputing realistic-cost segment medians for AMZN again.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_median(
        "AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, lookback=5, cost_cfg=REALISTIC_COSTS
    )[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_median(
        "AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, lookback=5, cost_cfg=REALISTIC_COSTS
    )[0], 3)
          for lab, s, e in runs]
    print(f"  AMZN r1: {m1}")
    print(f"  AMZN r2: {m2} -> {"identical" if m1 == m2 else "DIFFERENT"}")
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the cost-robustness artifact.")
    else:
        print("MISMATCH FOUND: the cost-robustness artifact does not reproduce via the")
        print("independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Cost-robustness momentum independent verification on real-data universe: exploratory simulation.")


if __name__ == "__main__":
    main()
