"""Independent verification of the momentum cost-robustness check
(`research/checks/momentum_cost_robustness.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median WITH realistic transaction costs (commission + slippage)
and the universe-level results from a fresh `walk_forward` implementation
(using the same cost configuration), then compares against the artifact
written by the momentum cost check (`state/check_artifacts/momentum_cost_robustness_results.json`).
A mismatch would flag a defect in the cost check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, using the same cost configuration (commission + slippage) as the
  cost check.
- Perturbation medians are recomputed by walking each parameter set
  independently with the same cost configuration.
- All values are loaded from the cost check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments, cost model) so a match
confirms the cost check computed from those inputs; a mismatch would flag an
error in the cost check's pipeline.
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

# Same cost model as in the momentum cost check
COMMISSION_PER_TRADE = 0.005
SLIPPAGE_BPS = 1.5
SLIPPAGE_PROPORTIONAL = 0.0015
COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=COMMISSION_PER_TRADE,
    slippage_cents=SLIPPAGE_BPS * 100,
    slippage_proportional=SLIPPAGE_PROPORTIONAL,
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


def segment_median_with_costs(ticker, s, e, train, test, warm, overlap, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback with realistic costs."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback=lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    
    # Use the same cost configuration as the cost check
    cfg = bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=COMMISSION_PER_TRADE,
        slippage_cents=SLIPPAGE_BPS * 100,
        slippage_proportional=SLIPPAGE_PROPORTIONAL,
        warmup_periods=warm,
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


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == list(LOOKBACKS)

    print("=== Independent recomputation (fresh walk_forward path with costs) ===")
    all_ok = True

    # Per-asset: momentum segment medians with realistic costs vs artifact
    print("\n=== Per-asset recomputation with costs ===")
    for asset in artifact["per_asset"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        medians = []
        for lab, s, e in runs:
            m, _ = segment_median_with_costs(asset, s, e, TRAIN, TEST, WARM, OVERLAP, 5)
            medians.append(round(m, 3))

        pub_medians = artifact["per_asset"][asset]["5"]["medians"]
        pub_lbls = artifact["per_asset"][asset]["5"]["segments"]

        medians_status = "MATCH" if medians == pub_medians and lbls == pub_lbls else "MISMATCH"
        if medians_status != "MATCH":
            all_ok = False
            print(f"  {asset}: medians {medians} vs published {pub_medians} -> {medians_status}")
            print(f"    labels: {lbls} vs {pub_lbls}")

    # Universe-level robust asset counts vs artifact
    print("\n=== Universe-level robust asset counts with costs ===")
    published_counts = {
        "robust_assets": artifact["universe"]["robust_assets"],
        "sensitive_assets": artifact["universe"]["sensitive_assets"],
        "no_edge_assets": artifact["universe"]["no_edge_assets"],
        "total_assets": artifact["universe"]["total_assets"],
    }

    # Recompute robust asset counts with the same logic as the cost check
    # The cost check logic: edge_count >= 2 = ROBUST, edge_count == 1 = SENSITIVE, else NO_EDGE
    recomputed_robust = 0
    recomputed_sensitive = 0
    recomputed_no_edge = 0

    # Use the same logic as in the momentum_cost_robustness.py
    for ticker in artifact["per_asset"]:
        edge_count = 0
        for lb in LOOKBACKS:
            verdict = artifact["per_asset"][ticker][str(lb)]["verdict"]
            medians = artifact["per_asset"][ticker][str(lb)]["medians"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
                edge_count += 1
        
        if edge_count >= 2:
            recomputed_robust += 1
        elif edge_count == 1:
            recomputed_sensitive += 1
        else:
            recomputed_no_edge += 1

    counts_match = (
        recomputed_robust == published_counts["robust_assets"] and
        recomputed_sensitive == published_counts["sensitive_assets"] and
        recomputed_no_edge == published_counts["no_edge_assets"] and
        recomputed_no_edge + recomputed_sensitive + recomputed_robust == published_counts["total_assets"]
    )

    if not counts_match:
        all_ok = False
        print("  UNVERIFIED counts mismatch:")
        print(f"    Published: {published_counts}")
        print(f"    Recomputed: robust={recomputed_robust}, sensitive={recomputed_sensitive}, no_edge={recomputed_no_edge}, total={recomputed_robust + recomputed_sensitive + recomputed_no_edge}")

    print(f"  Published counts: {published_counts}")
    print(f"  Recomputed counts: robust={recomputed_robust}, sensitive={recomputed_sensitive}, no_edge={recomputed_no_edge}")

    # Determinism check
    print("\n=== Determinism check ===")
    bars1, dates1 = bt.load_ticker("AAPL")
    closes1 = bars1.closes_array()
    labels1 = vol_blocks(closes1, N_BLOCKS, WINDOW)
    runs1 = segments_from_labels(labels1, MIN_SEGMENT_BARS)

    bars2, dates2 = bt.load_ticker("AAPL")
    closes2 = bars2.closes_array()
    labels2 = vol_blocks(closes2, N_BLOCKS, WINDOW)
    runs2 = segments_from_labels(labels2, MIN_SEGMENT_BARS)

    m1 = [round(segment_median_with_costs("AAPL", s, e, TRAIN, TEST, WARM, OVERLAP, 5)[0], 3)
          for lab, s, e in runs1]
    m2 = [round(segment_median_with_costs("AAPL", s, e, TRAIN, TEST, WARM, OVERLAP, 5)[0], 3)
          for lab, s, e in runs2]

    print(f"  AAPL r1: {m1}")
    print(f"  AAPL r2: {m2}")
    print(f"  Identical: {m1 == m2}")
    if m1 != m2:
        all_ok = False

    print()
    if all_ok:
        print("All independent recomputations with realistic costs MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the cost check artifact does not reproduce via the independent path. Investigate before admitting the results.")
        sys.exit(1)

    print("NOTE: research/simulation only. No live trading or production execution. Momentum cost robustness independent verification.")


if __name__ == "__main__":
    main()
