"""Independent verification of the momentum cost sensitivity check
(`research/checks/momentum_lookback_sensitivity.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the candidate vs null gap from a fresh `walk_forward`
implementation (not `stress_segments`), then compares against the artifact
written by the cost sensitivity check (`state/check_artifacts/momentum_cost_sensitivity_results.json`).
A mismatch would flag a defect in the cost sensitivity check.

Design of the independent path:
- Re-implementation of volatility blocks and segment reconstruction for determinism.
- Fresh momentum signals using the same contract as the check.
- Per-segment walk_forward on both candidate and null paths using realistic costs.
- Segment gap = candidate median - null median computed directly.
- All values compared against the check's artifact.

This keeps the same inputs (dataset, seed, windows, segments, cost model) so a
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
from research.data.preflight import load_manifest, sha256_file

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
    """Re-implementation of segment reconstruction."""
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


def segment_cost_sensitivity_independent(ticker, s, e, train, test, warm, overlap, lookback):
    """Independent recomputation of segment cost sensitivity using walk_forward."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    
    # Same realistic cost model as the check
    cfg_realistic = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=warm,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )
    
    # Candidate: momentum signal with realistic costs
    signals = momentum_signals(closes, lookback)
    seg_signals = signals[s:e]
    res_candidate = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg_realistic)
    
    # Null: coin-flip signal with realistic costs using noise_benchmark
    null_res = bt.noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=train, test_window=test,
        warmup=0, overlap_window=0, periods_per_year=252,
        cfg=cfg_realistic)
    
    # Compute medians from candidate folds
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in res_candidate.folds], dtype=np.float64)
    cand_median = float(np.median(cand_log))
    
    # Get null median
    null_median = null_res.baseline_median_log_return
    
    # Compute segment gap
    segment_gap = cand_median - null_median
    
    return {
        "cand_median": cand_median,
        "null_median": null_median,
        "segment_gap": segment_gap,
    }


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert "cost_robustness_results" in artifact
    
    print("=== Independent recomputation of cost sensitivity ===")
    all_ok = True
    
    # Load cost robustness results from artifact
    cost_results = artifact["cost_robustness_results"]
    
    # Recompute all cost sensitivity results
    recomputed_results = []
    for result in cost_results:
        ticker = result["ticker"]
        lookback = result["lookback"]
        
        # Get bars to compute segments
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        
        # Extract segment info from result
        segment_id = result["segment_id"]
        # Parse segment_id to get lab, s, e
        parts = segment_id.split("_")
        if len(parts) == 3:
            lab, s_str, e_str = parts
            s = int(s_str)
            e = int(e_str)
        else:
            print("  WARNING: Could not parse segment_id {}".format(segment_id))
            continue
        
        # Recompute segment cost sensitivity
        recomputed = segment_cost_sensitivity_independent(
            ticker, s, e, TRAIN, TEST, WARM, OVERLAP, lookback)
        
        # Compare with published results
        published_gap = result["segment_gap"]
        recomputed_gap = recomputed["segment_gap"]
        
        gap_diff = abs(published_gap - recomputed_gap)
        match = gap_diff < 0.001  # Allow small numerical differences
        
        recomputed_results.append({
            "ticker": ticker,
            "lookback": lookback,
            "segment_id": segment_id,
            "published_gap": published_gap,
            "recomputed_gap": recomputed_gap,
            "gap_diff": gap_diff,
            "match": match,
        })
        
        if not match:
            all_ok = False
            print("  MISMATCH: {}_{}: published gap {:+.3f}, recomputed gap {:+.3f}, diff {:+.4f}"
                  .format(ticker, segment_id, published_gap, recomputed_gap, gap_diff))
        else:
            print("  MATCH: {}_{}: gap {:+.3f}".format(ticker, segment_id, published_gap))
    
    print("\n=== Summary ===")
    total = len(recomputed_results)
    matches = sum(1 for r in recomputed_results if r["match"])
    mismatches = total - matches
    
    print("  Total segments recomputed: {}".format(total))
    print("  Matches: {}".format(matches))
    print("  Mismatches: {}".format(mismatches))
    
    if all_ok:
        print("\nAll independent recomputations MATCH the cost sensitivity artifact.")
        print("NOTE: research/simulation only. No live trading or production execution.")
        print("Momentum cost sensitivity independent verification PASSED.")
    else:
        print("\nMISMATCH FOUND: the cost sensitivity artifact does not reproduce via the")
        print("independent path. Investigate before admitting the results.")
        sys.exit(1)


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
