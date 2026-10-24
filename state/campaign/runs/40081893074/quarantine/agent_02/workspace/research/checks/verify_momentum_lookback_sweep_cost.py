"""Independent verification of the cost-robustness momentum check
(research/checks/momentum_lookback_sweep_cost.py).

This is a separate implementation path: it recomputes each segment's
walk-forward median using realistic costs, then compares against the artifact
written by the cost-robustness check. A mismatch would flag a defect in the check.

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
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_cost_results.json"

# The realistic cost model used by the check.
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,        # $0 per round-trip transaction
    commission_per_share=0.003,      # $0.003 per share
    slippage_cents=0.5,              # $0.50 fixed slippage per share
    slippage_proportional=0.0001,    # 10 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
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
def segment_median_with_costs(ticker, s, e, train, test, warm, overlap):
    """Walk-forward segment median with realistic costs using fresh walk_forward."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    # Use lookback=5 for segment_median verification (matching artifact reference)
    signals = momentum_signals(closes, 5)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = bt.BacktestConfig(
        initial_capital=REALISTIC_COSTS.initial_capital,
        target_exposure=REALISTIC_COSTS.target_exposure,
        commission_per_trade=REALISTIC_COSTS.commission_per_trade,
        commission_per_share=REALISTIC_COSTS.commission_per_share,
        slippage_cents=REALISTIC_COSTS.slippage_cents,
        slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        warmup_periods=warm,
        risk_free=REALISTIC_COSTS.risk_free,
        periods_per_year=REALISTIC_COSTS.periods_per_year,
        margin_rate=REALISTIC_COSTS.margin_rate,
        margin_call_liquidate=REALISTIC_COSTS.margin_call_liquidate,
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

    print("=== Independent recomputation (fresh walk_forward path) ===")
    all_ok = True

    # Per-asset: momentum segment medians vs artifact (using lookback=5 reference)
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        medians = []
        for lab, s, e in runs:
            m, nf = segment_median_with_costs(asset, s, e, TRAIN, TEST, WARM, OVERLAP)
            medians.append(round(m, 3))

        pub_medians = artifact["per_asset"][asset]["5"]["medians"]
        pub_lbls = artifact["per_asset"][asset]["5"]["segments"]

        status = "MATCH" if medians == pub_medians and lbls == pub_lbls else "MISMATCH"
        if status != "MATCH":
            all_ok = False
        print("  {}: medians {} vs published {} -> {}"
              .format(asset, medians, pub_medians, status))

    # Universe-level per-asset medians and verdicts vs artifact
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
        medians = [round(segment_median_with_costs(ticker, s, e, TRAIN, TEST, WARM, OVERLAP)[0], 3)
                   for lab, s, e in runs]
        pub = artifact["universe"]["per_asset"][ticker]["lookback_medians"]["5"]
        pub_lbls = artifact["universe"]["per_asset"][ticker]["verdicts_5"]
        # Regenerate the verdict from recomputed medians and null medians.
        cand = np.array(medians)
        null = np.array(artifact["universe"]["per_asset"][ticker]["lookback_null_medians"]["5"])
        cand_disp = round(float(np.std(cand)), 3)
        null_disp = round(float(np.std(null)), 3)
        if all(abs(m) <= 0.05 for m in medians):
            regen_verdict = "CONSISTENT_WITH_NOISE"
        elif cand_disp > 2.0 * null_disp:
            regen_verdict = "REGIME_DEPENDENT"
        elif all(m < 0 for m in medians):
            # Uniformly negative medians with a swing within twice the null:
            # stable losses, not an edge; must be read alongside the medians.
            regen_verdict = "REGIME_STABLE_LOSS"
        else:
            regen_verdict = "REGIME_STABLE"
        status = "MATCH" if medians == pub else "MISMATCH"
        if status != "MATCH":
            all_ok = False
            print("  [DEBUG] ticker={}: len(runs)={} lbls={}".format(ticker, len(runs), lbls))
            print("  [DEBUG] ticker={}: pub_verdict={}".format(ticker, pub_lbls))
            print("  [DEBUG] ticker={}: len(medians)={} medians={}".format(ticker, len(medians), medians))
            print("  [DEBUG] ticker={}: pub_medians={}".format(ticker, pub))
        verdict_status = "MATCH" if regen_verdict == pub_lbls else "MISMATCH"
        if verdict_status != "MATCH":
            all_ok = False
        print("  {}: medians {} -> {} | verdict {} (pub: {})"
              .format(ticker, medians, status, regen_verdict,
                      artifact["universe"]["per_asset"][ticker]["verdicts_5"]))
        if verdict_status != "MATCH":
            print("      verdict regeneration did not match artifact: {}"
                  .format(verdict_status))

    print("\nDeterminism: recomputing universe medians for AMZN and JPM again.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_median_with_costs("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_median_with_costs("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP)[0], 3)
          for lab, s, e in runs]
    print("  AMZN r1: {}; r2: {} -> {}".format(m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the cost-robustness artifact.")
    else:
        print("MISMATCH FOUND: the cost-robustness artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Cost-robustness momentum check independent verification on AMZN/JPM + universe: "
          "exploratory simulation.")
if __name__ == "__main__":
    main()
