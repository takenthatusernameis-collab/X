"""Independent verification of the momentum cost-robustness check
(`research/checks/momentum_cost_robustness.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median with zero and realistic costs, then compares against the
artifact written by the momentum cost-robustness check (`state/check_artifacts/
momentum_cost_robustness_results.json`). A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- Cost-sensitivity gaps are computed by running zero and realistic cost scenarios.
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

# Realistic transaction cost model (matching momentum_cost_robustness.py)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Zero-cost model for comparison
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
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
            labels.append(\"insufficient\")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window else 0.0)
            labels.append(\"turbulent\" if bmed > series_med else \"calm\")
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


def segment_median_costs(
    ticker, s, e, train, test, warm, overlap, cfg_zero, cfg_realistic
):
    """Walk-forward segment medians for both zero and realistic costs."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, LOOKBACK)

    # Zero cost scenario
    seg_bars_zero = list(bars)[s:e]
    seg_signals_zero = signals[s:e]
    cfg_zero_warm = bt.BacktestConfig(
        initial_capital=cfg_zero.initial_capital,
        commission_per_trade=cfg_zero.commission_per_trade,
        commission_per_share=cfg_zero.commission_per_share,
        slippage_cents=cfg_zero.slippage_cents,
        slippage_proportional=cfg_zero.slippage_proportional,
        warmup_periods=warm,
    )
    res_zero = bt.walk_forward(
        seg_bars_zero, seg_signals_zero, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg_zero_warm)
    zero_log = np.array([
        np.log1p(np.clip(f.metrics[\"total_return\"], -1.0 + 1e-12, None))
        for f in res_zero.folds], dtype=np.float64)

    # Realistic cost scenario
    seg_bars_real = list(bars)[s:e]
    seg_signals_real = signals[s:e]
    cfg_real_warm = bt.BacktestConfig(
        initial_capital=cfg_realistic.initial_capital,
        commission_per_trade=cfg_realistic.commission_per_trade,
        commission_per_share=cfg_realistic.commission_per_share,
        slippage_cents=cfg_realistic.slippage_cents,
        slippage_proportional=cfg_realistic.slippage_proportional,
        warmup_periods=warm,
    )
    res_real = bt.walk_forward(
        seg_bars_real, seg_signals_real, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg_real_warm)
    real_log = np.array([
        np.log1p(np.clip(f.metrics[\"total_return\"], -1.0 + 1e-12, None))
        for f in res_real.folds], dtype=np.float64)

    zero_median = float(np.median(zero_log)) if len(zero_log) > 0 else 0.0
    realistic_median = float(np.median(real_log)) if len(real_log) > 0 else 0.0

    return zero_median, realistic_median, len(res_zero.folds), len(res_real.folds)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact[\"dataset_id\"] == DATASET_ID, \"manifest id mismatch\"
    assert artifact[\"lookbacks\"] == list(LOOKBACKS)
    assert artifact[\"per_asset\"].keys() == {\"AMZN\", \"JPM\"}

    print("=== Independent recomputation (cost-robustness check) ===")
    all_ok = True

    # Per-asset: recompute zero and realistic cost medians for lookback=5
    for asset in [\"AMZN\", \"JPM\"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        zero_medians = []
        realistic_medians = []
        for lab, s, e in runs:
            zero_med, realistic_med, _, _ = segment_median_costs(
                asset, s, e, TRAIN, TEST, WARM, OVERLAP, ZERO_COSTS, REALISTIC_COSTS)
            zero_medians.append(round(zero_med, 3))
            realistic_medians.append(round(realistic_med, 3))

        # Compare against artifact
        pub_zero = artifact[\"per_asset\"][\"5\"][asset][\"zero_medians\"]
        pub_realistic = artifact[\"per_asset\"][\"5\"][asset][\"realistic_medians\"]

        zero_status = \"MATCH\" if zero_medians == pub_zero and lbls == artifact[\"per_asset\"][\"5\"][asset][\"segments\"] else \"MISMATCH\"
        realistic_status = \"MATCH\" if realistic_medians == pub_realistic and lbls == artifact[\"per_asset\"][\"5\"][asset][\"segments\"] else \"MISMATCH\"

        if zero_status != \"MATCH\" or realistic_status != \"MATCH\":
            all_ok = False

        print(f\"  {asset}: zero_cost medians {zero_medians} vs published {pub_zero} -> {zero_status}\")
        print(f\"  {asset}: realistic_cost medians {realistic_medians} vs published {pub_realistic} -> {realistic_status}\")

    # Universe-level cost-robustness verdict verification
    print(\"\n=== Universe-level recomputation ===\")
    tickers = {}
    from research.data.preflight import load_manifest
    m = load_manifest()
    for entry in m[\"entries\"]:
        tickers[entry[\"ticker\"]] = bt.load_ticker(entry[\"ticker\"])[0]

    # Check cost-robustness verdicts
    robustness_counts = {\"ROBUST\": 0, \"SENSITIVE\": 0, \"NO_EDGE\": 0}
    for ticker in artifact[\"universe\"][\"per_asset\"]:
        pub_zero_verdict = artifact[\"universe\"][\"per_asset\"][ticker][\"zero_verdict\"]
        pub_realistic_verdict = artifact[\"universe\"][\"per_asset\"][ticker][\"realistic_verdict\"]
        pub_robustness = artifact[\"universe\"][\"per_asset\"][ticker][\"robustness_verdict\"]

        # Recompute from medina
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)

        # Collect zero and realistic medians for lookback=5
        zero_medians = []
        realistic_medians = []
        for lab, s, e in runs:
            zero_med, realistic_med, _, _ = segment_median_costs(
                ticker, s, e, TRAIN, TEST, WARM, OVERLAP, ZERO_COSTS, REALISTIC_COSTS)
            zero_medians.append(zero_med)
            realistic_medians.append(realistic_med)

        # Determine verdicts
        zero_verdict = \"CONSISTENT_WITH_NOISE\"
        if all(abs(m) > 0.05 for m in zero_medians):
            if all(m > 0 for m in zero_medians):
                zero_verdict = \"REGIME_STABLE\"
            elif all(m < 0 for m in zero_medians):
                zero_verdict = \"REGIME_STABLE_LOSS\"

        realistic_verdict = \"CONSISTENT_WITH_NOISE\"
        if all(abs(m) > 0.05 for m in realistic_medians):
            if all(m > 0 for m in realistic_medians):
                realistic_verdict = \"REGIME_STABLE\"
            elif all(m < 0 for m in realistic_medians):
                realistic_verdict = \"REGIME_STABLE_LOSS\"

        # Cost-robustness verdict
        edge_counts_zero = {\"EDGE\": 0, \"NO_EDGE\": 0}
        for lb in LOOKBACKS:
            if lb == 5 and zero_verdict == \"REGIME_STABLE\" and all(m > 0 for m in zero_medians):
                edge_counts_zero[\"EDGE\"] += 1
            else:
                edge_counts_zero[\"NO_EDGE\"] += 1

        robust_gaps = True
        if edge_counts_zero[\"EDGE\"] >= 2:
            for zero_med, realistic_med in zip(zero_medians, realistic_medians):
                if (zero_med - realistic_med) <= 0:
                    robust_gaps = False
                    break
        else:
            robust_gaps = False

        if edge_counts_zero[\"EDGE\"] >= 2 and robust_gaps:
            regen_robustness = \"ROBUST\"
        elif edge_counts_zero[\"EDGE\"] == 1:
            regen_robustness = \"SENSITIVE\"
        else:
            regen_robustness = \"NO_EDGE\"

        # Check against published verdicts
        zero_ok = zero_verdict == pub_zero_verdict
        realistic_ok = realistic_verdict == pub_realistic_verdict
        robustness_ok = regen_robustness == pub_robustness

        if not zero_ok or not realistic_ok or not robustness_ok:
            all_ok = False

        print(f\"  {ticker}: zero_verdict {zero_verdict} (pub: {pub_zero_verdict}) -> {\"MATCH\" if zero_ok else \"MISMATCH\"}\")
        print(f\"  {ticker}: realistic_verdict {realistic_verdict} (pub: {pub_realistic_verdict}) -> {\"MATCH\" if realistic_ok else \"MISMATCH\"}\")
        print(f\"  {ticker}: robustness {regen_robustness} (pub: {pub_robustness}) -> {\"MATCH\" if robustness_ok else \"MISMATCH\"}\")

    print()
    if all_ok:
        print(\"All independent recomputations MATCH the cost-robustness artifact.\")
    else:
        print(\"MISMATCH FOUND: the cost-robustness artifact does not reproduce via the independent path. Investigate before admitting the results.\")
        sys.exit(1)
    print(\"NOTE: research/simulation only. No live trading or production execution. Momentum cost-robustness independent verification on AMZN/JPM + universe: exploratory simulation.\")


if __name__ == \"__main__\":
    main()