#!/usr/bin/env python3
"""Simple verification script for cost sensitivity momentum check - using framework's noise_benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest
from research.backtest.perturbation import noise_benchmark

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

REALISTIC_COST_CONFIG = bt.BacktestConfig(
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    commission_per_trade=2.0,
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
    """Re-implementation of segment extraction."""
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
    """Fresh implementation of momentum signal."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def main():
    np.random.seed(42)
    ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "cost_sensitivity_results.json"
    
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    
    assert artifact["dataset_id"] == DATASET_ID
    
    print("=== Independent verification of cost sensitivity momentum check ===")
    print("Verifying that the check's results are reproducible using framework's noise_benchmark...")
    
    # Load manifest
    manifest = load_manifest()
    
    all_ok = True
    
    # Verify each lookback
    for lb in LOOKBACKS:
        print(f"\n--- Lookback {lb} days ---")
        lb_key = f"lookback_{lb}"
        
        if lb_key not in artifact["results"]:
            print(f"  ERROR: No results found for lookback {lb}")
            all_ok = False
            continue
        
        artifact_result = artifact["results"][lb_key]
        
        # Verify each asset
        for ticker in artifact_result["per_asset"]:
            artifact_asset = artifact_result["per_asset"][ticker]
            
            # Reconstruct the same data
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            
            # Recompute for each segment
            recomputed_medians = []
            recomputed_null_medians = []
            for lab, s, e in runs:
                # Candidate with costs (same as check)
                bars_seg, dates_seg = bt.load_ticker(ticker)
                closes_seg = bars_seg.closes_array()[s:e]
                signals = momentum_signals(closes_seg, lb)
                result = bt.walk_forward(
                    list(bars_seg)[s:e], signals, train_window=TRAIN, test_window=TEST,
                    warmup=WARM, overlap_window=OVERLAP, cfg=REALISTIC_COST_CONFIG)
                log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                                for f in result.folds], dtype=np.float64)
                cand_med = float(np.median(log))
                recomputed_medians.append(round(cand_med, 3))
                
                # Null with costs (using framework's noise_benchmark)
                noise = noise_benchmark(
                    bars=list(bars_seg)[s:e],
                    param_grid=[{"lookback": lb}],
                    train_window=TRAIN, test_window=TEST,
                    warmup=0, overlap_window=0, periods_per_year=252,
                    cfg=REALISTIC_COST_CONFIG)
                null_med = noise.baseline_median_log_return
                recomputed_null_medians.append(round(null_med, 3))
            
            # Compare with artifact
            artifact_medians = artifact_asset["medians"]
            artifact_null_medians = artifact_asset["null_medians"]
            
            med_match = recomputed_medians == artifact_medians
            null_match = recomputed_null_medians == artifact_null_medians
            
            if med_match and null_match:
                print(f"  {ticker}: MATCH")
            else:
                all_ok = False
                print(f"  {ticker}: MISMATCH")
                if not med_match:
                    print(f"    Candidate medians recomputed: {recomputed_medians}")
                    print(f"    Artifact: {artifact_medians}")
                if not null_match:
                    print(f"    Null medians recomputed: {recomputed_null_medians}")
                    print(f"    Artifact: {artifact_null_medians}")
        
        print(f"  Lookback {lb}: {'VERIFIED' if all_ok else 'MISMATCH'}")
    
    print("\n=== Verification Summary ===")
    if all_ok:
        print("SUCCESS: All independent recomputations MATCH the cost sensitivity check artifact.")
        print("The momentum cost-robustness results are reproducible and independently verified.")
    else:
        print("FAILURE: MISMATCH FOUND. The cost sensitivity artifact does not reproduce via the independent path.")
        print("Investigate before admitting the results.")
        sys.exit(1)

if __name__ == "__main__":
    main()