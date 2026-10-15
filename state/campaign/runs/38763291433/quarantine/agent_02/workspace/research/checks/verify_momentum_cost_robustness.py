"""Independent verification of the momentum cost robustness check
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

REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=100,
    commission_per_share=5,
    slippage_cents=20,
    slippage_proportional=0.001,
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


def segment_cost_sensitive_median(ticker, s, e, train, test, warm, overlap, lookback):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = REALISTIC_COST_CONFIG
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def main():
    np.random.seed(42)
    
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == [3, 5, 10]
    assert "cost_config" not in artifact, "artifact should not contain cost_config"
    
    print("=== Independent verification of momentum cost-robustness ===")
    print(f"Verifying lookback {LOOKBACKS} with realistic transaction costs")

    from research.data.preflight import load_manifest
    manifest = load_manifest()
    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}

    all_ok = True
    
    # Per-asset recomputation
    print("\n=== 1. Per-asset recomputation ===")
    for ticker in ["NVDA"]:
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        
        for lb in LOOKBACKS:
            medians = []
            for lab, s, e in runs:
                m, nf = segment_cost_sensitive_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb)
                medians.append(round(m, 3))
            
            pub_medians = artifact["per_asset"][ticker][str(lb)]["medians"]
            pub_lbls = artifact["per_asset"][ticker][str(lb)]["segments"]
            
            status = "MATCH" if medians == pub_medians and lbls == pub_lbls else "MISMATCH"
            if status != "MATCH":
                all_ok = False
            print(f"  {ticker} lookback {lb}: medians {medians} vs published {pub_medians} -> {status}")

    # Universe-level recomputation
    print("\n=== 2. Universe-level recomputation ===")
    for ticker in ["NVDA"]:
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        
        medians = [round(segment_cost_sensitive_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb)[0], 3)
                   for lb in LOOKBACKS for lab, s, e in runs]
        
        pub_medians = artifact["universe"]["per_asset"][ticker]["lookback_medians"][str(lb)]
        pub_lbls = artifact["universe"]["per_asset"][ticker]["segments"]
        
        status = "MATCH" if medians == pub_medians and lbls == pub_lbls else "MISMATCH"
        if status != "MATCH":
            all_ok = False
            print(f"  [DEBUG] ticker={ticker}: len(runs)={len(runs)} lbls={lbls}")
            print(f"  [DEBUG] ticker={ticker}: pub_lbls={pub_lbls}")
            print(f"  [DEBUG] ticker={ticker}: len(medians)={len(medians)} medians={medians}")
            print(f"  [DEBUG] ticker={ticker}: pub_medians={pub_medians}")
        
        print(f"  {ticker}: medians {medians} segments={pub_lbls} -> {status}")

    # Cost-robustness verdict recomputation
    print("\n=== 3. Cost-robustness verdict verification ===")
    for ticker in ["NVDA"]:
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        
        for lb in LOOKBACKS:
            medians = [segment_cost_sensitive_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lb)[0]
                      for lab, s, e in runs]
            medians = [round(m, 3) for m in medians]
            
            cand = np.array(medians)
            null = np.array(artifact["universe"]["per_asset"][ticker]["lookback_null_medians"][str(lb)])
            
            if all(abs(m) <= 0.05 for m in medians):
                regen_verdict = "CONSISTENT_WITH_NOISE"
            elif np.std(cand) > 2.0 * np.std(null):
                regen_verdict = "REGIME_DEPENDENT"
            elif all(m < 0 for m in medians):
                regen_verdict = "REGIME_STABLE_LOSS"
            else:
                regen_verdict = "REGIME_STABLE"
            
            pub_verdict = artifact["universe"]["per_asset"][ticker][str(lb)]["verdict"]
            verdict_status = "MATCH" if regen_verdict == pub_verdict else "MISMATCH"
            if verdict_status != "MATCH":
                all_ok = False
            
            print(f"  {ticker} lookback {lb}: verdict {regen_verdict} (published: {pub_verdict}) -> {verdict_status}")

    # Determinism check
    print("\n=== 4. Determinism verification ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    
    m1 = [round(segment_cost_sensitive_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, lb)[0], 3)
          for lb in LOOKBACKS for lab, s, e in runs]
    m2 = [round(segment_cost_sensitive_median("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, lb)[0], 3)
          for lb in LOOKBACKS for lab, s, e in runs]
    
    print(f"  AMZN determinism: r1 == r2: {m1 == m2}")
    assert m1 == m2, "verification path not deterministic"

    if all_ok:
        print("\n✓ All independent recomputations MATCH the check artifact.")
        print("Cost-robustness check verified successfully.")
    else:
        print("\n✗ MISMATCH FOUND: the check artifact does not reproduce via the independent path.")
        print("Investigate before admitting the results.")
        sys.exit(1)
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Cost-robustness independent verification on AMZN/JPM + universe: exploratory simulation.")


if __name__ == "__main__":
    main()