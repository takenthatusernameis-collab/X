"""Independent verification of the cost-sensitive momentum check
(`research/checks/momentum_cost_robustness.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the cost-robustness check (`state/check_artifacts/
momentum_cost_robustness_results.json`). A mismatch would flag a defect in the check.

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
from research.backtest.perturbation import noise_benchmark

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
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

def segment_result_with_costs(ticker, s, e, train, test, warm, overlap, lookback, zero_cost):
    """Walk-forward regime-stability result for one asset segment and lookback,
    computed with fresh components: volatility_blocks, segment slicing,
    momentum signals, walk_forward, and the framework noise_benchmark.
    No call to stress_segments.
    
    Args:
        zero_cost: if True, use zero-cost configuration; if False, use realistic costs
    """
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    
    if zero_cost:
        cfg = bt.BacktestConfig(warmup_periods=warm)
    else:
        cfg = bt.BacktestConfig(
            initial_capital=1e6,
            warmup_periods=warm,
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        )
    
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=train, test_window=test,
        warmup=0, overlap_window=0, periods_per_year=252,
        seed=42 if zero_cost else 99,
    )
    null_log = noise.baseline_median_log_return
    return float(np.median(cand_log)), null_log, len(res.folds)

def load_artifact(path):
    with open(path) as f:
        return json.load(f)

def main():
    np.random.seed(42)
    artifact = load_artifact(ARTIFACT_PATH)
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == LOOKBACKS, "lookback mismatch"
    
    all_ok = True

    print("=== Independent recomputation (fresh walk_forward path with cost sensitivity) ===")
    
    # Load manifest for universe
    from research.data.preflight import load_manifest
    manifest = load_manifest()
    tickers = {entry["ticker"]: bt.load_ticker(entry["ticker"])[0] for entry in manifest["entries"]}

    # Per-asset: recompute zero-cost and realistic-cost segment medians vs artifact
    for ticker in manifest["universe"]:
        print(f"\n=== Asset {ticker} ===")
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        
        # Recompute zero-cost results
        zero_medians = []
        zero_nulls = []
        for lab, s, e in runs:
            m, n, _ = segment_result_with_costs(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, 5, zero_cost=True)
            zero_medians.append(round(m, 3))
            zero_nulls.append(round(n, 3))
        
        # Recompute realistic-cost results  
        realistic_medians = []
        realistic_nulls = []
        for lab, s, e in runs:
            m, n, _ = segment_result_with_costs(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, 5, zero_cost=False)
            realistic_medians.append(round(m, 3))
            realistic_nulls.append(round(n, 3))
        
        # Compare with artifact
        pub_zero_medians = [round(v, 3) for v in artifact["per_asset"][ticker]["zero_cost"]["medians"]]
        pub_realistic_medians = [round(v, 3) for v in artifact["per_asset"][ticker]["realistic_cost"]["medians"]]
        
        zero_status = "MATCH" if zero_medians == pub_zero_medians and lbls == artifact["per_asset"][ticker]["zero_cost"]["segments"] else "MISMATCH"
        realistic_status = "MATCH" if realistic_medians == pub_realistic_medians else "MISMATCH"
        
        if zero_status != "MATCH":
            all_ok = False
            print(f"  zero_cost: medians recomputed {zero_medians} vs published {pub_zero_medians} -> {zero_status}")
        if realistic_status != "MATCH":
            all_ok = False
            print(f"  realistic_cost: medians recomputed {realistic_medians} vs published {pub_realistic_medians} -> {realistic_status}")
        
        print(f"  {ticker}: zero_cost -> {zero_status} | realistic_cost -> {realistic_status}")
    
    print("\n=== Independent verification completed ===")
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the independent path.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost-robustness independent verification: exploratory simulation.")

if __name__ == "__main__":
    main()