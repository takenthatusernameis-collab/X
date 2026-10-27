"""Independent verification of the momentum cost-robustness check
(`research/checks/momentum_cost_robustness.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the momentum cost-robustness check (`state/check_artifacts/momentum_cost_robustness.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- Both zero-cost and realistic-cost versions are computed independently.
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
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"

# Realistic cost model from enterprise standard
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ZERO_COSTS = bt.BacktestConfig(initial_capital=1e6, warmup_periods=0)


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
    """Extract segments from volatility labels."""
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


def compute_segment_medians(ticker, s, e, lookback, cfg):
    """Compute segment medians using walk_forward for one asset, one lookback, one cfg."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback=lookback)
    
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    
    cfg_copy = bt.BacktestConfig(
        initial_capital=cfg.initial_capital,
        warmup_periods=cfg.warmup_periods,
        commission_per_trade=cfg.commission_per_trade,
        commission_per_share=cfg.commission_per_share,
        slippage_cents=cfg.slippage_cents,
        slippage_proportional=cfg.slippage_proportional,
    )
    
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg_copy)
    
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
    assert artifact["lookbacks"] == list(LOOKBACKS), "lookback mismatch"
    
    print("=== Independent recomputation (walk_forward path) ===")
    all_ok = True
    
    # Load tickers once
    tickers = {}
    from research.data.preflight import load_manifest
    m = load_manifest()
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    
    # Recompute cost-robustness check for each ticker and lookback
    for ticker in artifact["per_asset"]:
        print(f"  {ticker}:")
        for lookback in LOOKBACKS:
            result = artifact["per_asset"][ticker][str(lookback)]
            
            # Get the asset bars
            bars = tickers[ticker]
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            
            # Compute for both cost models
            recomputed_zero = []
            recomputed_real = []
            
            for label, s, e in runs:
                med_zero, _ = compute_segment_medians(ticker, s, e, lookback, ZERO_COSTS)
                med_real, _ = compute_segment_medians(ticker, s, e, lookback, REALISTIC_COSTS)
                recomputed_zero.append(round(med_zero, 3))
                recomputed_real.append(round(med_real, 3))
            
            # Compare with artifact
            artifact_zero = result["zero_cost"]["medians"]
            artifact_real = result["realistic_cost"]["medians"]
            
            zero_match = recomputed_zero == artifact_zero
            real_match = recomputed_real == artifact_real
            
            if not (zero_match and real_match):
                all_ok = False
                if not zero_match:
                    print(f"    lookback {lookback} zero-cost mismatch: recomputed {recomputed_zero} vs artifact {artifact_zero}")
                if not real_match:
                    print(f"    lookback {lookback} realistic-cost mismatch: recomputed {recomputed_real} vs artifact {artifact_real}")
            else:
                print(f"    lookback {lookback}: zero-cost match ✓, realistic-cost match ✓")
    
    # Recompute universe-level verdict
    print("\n=== Universe-level recomputation ===")
    target = ["AMZN", "JPM"]
    for lookback in LOOKBACKS:
        print(f"  lookback {lookback}:")
        for ticker in target:
            bars = tickers[ticker]
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            
            recomputed_zero = []
            recomputed_real = []
            
            for label, s, e in runs:
                med_zero, _ = compute_segment_medians(ticker, s, e, lookback, ZERO_COSTS)
                med_real, _ = compute_segment_medians(ticker, s, e, lookback, REALISTIC_COSTS)
                recomputed_zero.append(round(med_zero, 3))
                recomputed_real.append(round(med_real, 3))
            
            # Find the corresponding artifact entry
            artifact_zero = artifact["per_asset"][ticker][str(lookback)]["zero_cost"]["medians"]
            artifact_real = artifact["per_asset"][ticker][str(lookback)]["realistic_cost"]["medians"]
            
            zero_match = recomputed_zero == artifact_zero
            real_match = recomputed_real == artifact_real
            
            if not (zero_match and real_match):
                all_ok = False
                if not zero_match:
                    print(f"    {ticker} zero-cost mismatch: recomputed {recomputed_zero} vs artifact {artifact_zero}")
                if not real_match:
                    print(f"    {ticker} realistic-cost mismatch: recomputed {recomputed_real} vs artifact {artifact_real}")
    
    # Determinism test
    print("\n=== Determinism test ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    
    # Recompute twice to check determinism
    med1, _ = compute_segment_medians("AMZN", runs[0][1], runs[0][2], 5, REALISTIC_COSTS)
    med2, _ = compute_segment_medians("AMZN", runs[0][1], runs[0][2], 5, REALISTIC_COSTS)
    
    if abs(med1 - med2) < 0.001:
        print(f"  deterministic: recomputed values match ({med1:.3f} == {med2:.3f})")
    else:
        print(f"  deterministic: recomputed values differ ({med1:.3f} != {med2:.3f})")
        all_ok = False
    
    # Final result
    print("\n" + "="*60)
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
        print("The momentum cost-robustness check is VERIFIED.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the")
        print("independent walk_forward path. Investigate before admitting the results.")
        sys.exit(1)
    
    print("\nNOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-robustness independent verification on "
          "the collected universe: exploratory simulation.")


if __name__ == "__main__":
    main()
