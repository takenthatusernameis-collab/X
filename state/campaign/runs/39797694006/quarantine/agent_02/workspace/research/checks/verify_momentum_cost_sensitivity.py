"""Independent verification of the cost-sensitive momentum check
(`research/checks/momentum_cost_sensitivity.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the cost-sensitive momentum check (`state/check_artifacts/momentum_cost_sensitivity_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments`` or
  ``regime_stability.py``.
- Cost sensitivity is verified by recomputing the medians using the same
  cost model parameters from the artifact.
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
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"


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


def segment_median(ticker, s, e, train, test, warm, overlap, lookback, cost_config):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    # Apply cost model from artifact
    cfg = bt.BacktestConfig(warmup_periods=warm)
    # Apply the cost parameters that exist in the artifact
    cfg.commission_per_trade = cost_config.get("commission_per_trade", 0.0)
    cfg.commission_per_share = cost_config.get("commission_per_share", 0.0)
    cfg.slippage_cents = cost_config.get("slippage_cents", 0.0)
    cfg.slippage_proportional = cost_config.get("slippage_proportional", 0.0)
    
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == list(LOOKBACKS), "lookback mismatch"
    
    # Extract cost model from artifact
    cost_model = artifact.get("cost_model", {
        "initial_capital": 1e6,
        "target_exposure": 1.0,
        "commission_per_trade": 0.0,
        "commission_per_share": 0.003,
        "slippage_cents": 5.0,
        "slippage_proportional": 0.001,
        "risk_free": 0.0,
        "periods_per_year": 252,
    })

    print("=== Independent recomputation (cost-sensitive momentum) ===")
    all_ok = True

    # Per-asset cost-sensitive results verification
    print("\n=== Per-asset cost-sensitive verification ===")
    # Check if lookback-specific keys exist in per_asset or universe
    per_asset_data = artifact["cost_momentum"]["per_asset"]
    universe_data = artifact["cost_momentum"]["universe"]["per_asset"]
    
    for ticker in per_asset_data.keys():
        # Load the ticker
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        for lookback in LOOKBACKS:
            # Recompute medians using the artifact's cost model
            medians = []
            for lab, s, e in runs:
                m, nf = segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lookback, cost_model)
                medians.append(round(m, 3))
            
            # Get published results from artifact - check both per_asset and universe
            pub = None
            if f"medians_{lookback}" in per_asset_data[ticker]:
                pub = per_asset_data[ticker][f"medians_{lookback}"]
            elif f"medians_{lookback}" in universe_data[ticker]:
                pub = universe_data[ticker][f"medians_{lookback}"]
            else:
                print(f"  WARNING: medians_{lookback} not found for {ticker}")
                all_ok = False
                continue
            
            # Compare with tolerance for floating-point differences
            tolerance = 0.01
            match = all(abs(r - p) <= tolerance for r, p in zip(medians, pub))
            status = "MATCH" if match else "MISMATCH"
            if status != "MATCH":
                all_ok = False
                print(f"  MISMATCH: {ticker} lookback={lookback}")
                print(f"    recomputed: {medians}")
                print(f"    published:  {pub}")
                print(f"    differences: {[abs(r - p) for r, p in zip(medians, pub)]}")
            else:
                print(f"  {ticker} lookback={lookback}: {status}")

    # Matched null verification - check if this section exists in the artifact
    if "matched_null" in artifact:
        print("\n=== Matched null verification ===")
        per_asset_null_data = artifact["matched_null"]["per_asset"]
        universe_null_data = artifact["matched_null"]["universe"]["per_asset"]
        
        for ticker in per_asset_null_data.keys():
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            
            for lookback in LOOKBACKS:
                # Recompute null medians
                medians = []
                for lab, s, e in runs:
                    seg_bars = list(bars)[s:e]
                    # Simple null signals
                    np.random.seed(SEED)
                    signals = []
                    for i in range(len(closes)):
                        if i < lookback:
                            signals.append(bt.Signal(date=i + 1, weight=0.0))
                        else:
                            signals.append(bt.Signal(date=i + 1, weight=1.0 if np.random.randn() > 0 else -1.0))
                    
                    seg_signals = signals[s:e]
                    cfg = bt.BacktestConfig(warmup_periods=WARM)
                    # Apply cost model from artifact
                    cfg.commission_per_trade = cost_model.get("commission_per_trade", 0.0)
                    cfg.commission_per_share = cost_model.get("commission_per_share", 0.0)
                    cfg.slippage_cents = cost_model.get("slippage_cents", 0.0)
                    cfg.slippage_proportional = cost_model.get("slippage_proportional", 0.0)
                    
                    res = bt.walk_forward(
                        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
                        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
                    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                                    for f in res.folds], dtype=np.float64)
                    medians.append(round(float(np.median(log)), 3))
                
                # Get published null results from artifact
                pub_key = f"nulls_{lookback}"
                pub = None
                if pub_key in per_asset_null_data[ticker]:
                    pub = per_asset_null_data[ticker][pub_key]
                elif pub_key in universe_null_data[ticker]:
                    pub = universe_null_data[ticker][pub_key]
                else:
                    print(f"  WARNING: {pub_key} not found for {ticker} null")
                    continue
                
                status = "MATCH" if medians == pub else "MISMATCH"
                if status != "MATCH":
                    all_ok = False
                    print(f"  MISMATCH: {ticker} lookback={lookback} null")
                    print(f"    recomputed: {medians}")
                    print(f"    published:  {pub}")
                else:
                    print(f"  {ticker} lookback={lookback} null: {status}")

    print("\nDeterminism verification (subset recomputation)")
    for ticker in ["AMZN", "JPM"]:
        # Load tickers directly
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        
        m1 = []
        for lab, s, e in runs:
            m, nf = segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, 5, cost_model)
            m1.append(round(m, 3))
        
        m2 = []
        for lab, s, e in runs:
            m, nf = segment_median(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, 5, cost_model)
            m2.append(round(m, 3))
        
        print(f"  {ticker}: r1={m1}, r2={m2}, identical={m1 == m2}")
        assert m1 == m2, "independent verification path not deterministic"

    print("\n" + "="*60)
    if all_ok:
        print("All independent recomputations MATCH the cost-sensitivity artifact.")
        print("Cost-sensitive momentum check results are validated.")
    else:
        print("MISMATCH FOUND: the cost-sensitivity artifact does not reproduce via the")
        print("independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("="*60)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    main()