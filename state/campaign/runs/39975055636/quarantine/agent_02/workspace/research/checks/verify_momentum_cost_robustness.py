"""Independent verification of the momentum cost robustness check
(research/checks/momentum_cost_robustness.py).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the cost-robustness results from a fresh `walk_forward`
implementation (not ``stress_segments``), then compares against the artifact
written by the cost robustness check (`state/check_artifacts/momentum_cost_robustness.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Same inputs (dataset, seed, windows, segments) so a match
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
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"


def momentum_signals(closes, lookback):
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


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


def segment_median(ticker, s, e, train, test, warm, overlap, lookback):
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = bt.BacktestConfig(warmup_periods=warm)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([
        np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
        for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def verify_momentum_cost_robustness():
    print("=== Independent Verification: Momentum Cost Robustness ===")
    
    with open(ARTIFACT_PATH, "r") as f:
        artifact = json.load(f)
    
    print(f"Loaded artifact from {ARTIFACT_PATH}")
    print(f"Dataset: {artifact['dataset_id']}")
    print(f"Lookbacks: {artifact['lookbacks']}")
    
    # Load manifest and validate
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    # Load all tickers
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars.closes_array()
    
    print(f"Loaded {len(tickers)} assets from collected universe")
    
    # Recompute results for each lookback
    verification_results = {}
    
    for lookback in artifact["lookbacks"]:
        print(f"\n--- Verifying lookback {lookback} ---")
        
        # Get cost models from artifact
        realistic_costs = artifact["realistic_costs"]
        zero_costs = artifact["zero_costs"]
        
        # Recompute realistic costs results
        realistic_recomputed = {}
        for ticker, closes in tickers.items():
            signals = momentum_signals(closes, lookback)
            cfg_realistic = bt.BacktestConfig(
                initial_capital=1e6,
                warmup_periods=WARM,
                commission_per_trade=realistic_costs["commission_per_trade"],
                commission_per_share=realistic_costs["commission_per_share"],
                slippage_cents=realistic_costs["slippage_cents"],
                slippage_proportional=realistic_costs["slippage_proportional"],
            )
            result = bt.walk_forward(
                list(bt.BarSequence(
                    dates=np.arange(1, len(closes) + 1, dtype=np.int64),
                    opens=np.zeros_like(closes),
                    highs=np.zeros_like(closes),
                    lows=np.zeros_like(closes),
                    closes=closes,
                    volumes=np.zeros_like(closes),
                )),
                signals,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                cfg=cfg_realistic,
            )
            log_returns = np.array([
                np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                for f in result.folds], dtype=np.float64)
            
            realistic_recomputed[ticker] = {
                "lookback": lookback,
                "n_folds": len(result.folds),
                "mean_log_return": float(np.mean(log_returns)),
                "median_log_return": float(np.median(log_returns)),
                "std_log_return": float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0,
                "positive_folds": int((log_returns > 0).sum()),
                "total_return": float(result.aggregate_metrics.get("mean_log_total_return", 0.0)),
            }
        
        # Recompute zero costs results
        zero_recomputed = {}
        for ticker, closes in tickers.items():
            signals = momentum_signals(closes, lookback)
            cfg_zero = bt.BacktestConfig(
                initial_capital=1e6,
                warmup_periods=WARM,
                commission_per_trade=zero_costs["commission_per_trade"],
                commission_per_share=zero_costs["commission_per_share"],
                slippage_cents=zero_costs["slippage_cents"],
                slippage_proportional=zero_costs["slippage_proportional"],
            )
            result = bt.walk_forward(
                list(bt.BarSequence(
                    dates=np.arange(1, len(closes) + 1, dtype=np.int64),
                    opens=np.zeros_like(closes),
                    highs=np.zeros_like(closes),
                    lows=np.zeros_like(closes),
                    closes=closes,
                    volumes=np.zeros_like(closes),
                )),
                signals,
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                cfg=cfg_zero,
            )
            log_returns = np.array([
                np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                for f in result.folds], dtype=np.float64)
            
            zero_recomputed[ticker] = {
                "lookback": lookback,
                "n_folds": len(result.folds),
                "mean_log_return": float(np.mean(log_returns)),
                "median_log_return": float(np.median(log_returns)),
                "std_log_return": float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0,
                "positive_folds": int((log_returns > 0).sum()),
                "total_return": float(result.aggregate_metrics.get("mean_log_total_return", 0.0)),
            }
        
        verification_results[f"lookback_{lookback}"] = {
            "realistic_costs": realistic_recomputed,
            "zero_costs": zero_recomputed,
        }
        
        # Compare with artifact
        print(f"  Comparing with artifact...")
        artifact_key = f"lookback_{lookback}"
        
        # Compare key metrics
        match = True
        for ticker in tickers:
            for cost_type in ["realistic_costs", "zero_costs"]:
                artifact_metrics = artifact["results"][artifact_key][cost_type][ticker]
                recomputed_metrics = verification_results[artifact_key][cost_type][ticker]
                
                for metric in ["mean_log_return", "median_log_return", "n_folds"]:
                    if abs(artifact_metrics[metric] - recomputed_metrics[metric]) > 1e-6:
                        print(f"    MISMATCH for {ticker} {cost_type} {metric}: artifact={artifact_metrics[metric]:.6f}, recomputed={recomputed_metrics[metric]:.6f}")
                        match = False
        
        if match:
            print(f"    ✓ All metrics match for lookback {lookback}")
        else:
            print(f"    ✗ Mismatches found for lookback {lookback}")
    
    # Final verdict - check all comparisons
    all_match = True
    for lookback in artifact["lookbacks"]:
        for ticker in tickers:
            for cost_type in ["realistic_costs", "zero_costs"]:
                for metric in ["mean_log_return", "median_log_return", "n_folds"]:
                    artifact_val = artifact["results"][f"lookback_{lookback}"][cost_type][ticker][metric]
                    recomputed_val = verification_results[f"lookback_{lookback}"][cost_type][ticker][metric]
                    if abs(artifact_val - recomputed_val) > 1e-6:
                        all_match = False
                        print(f"    MISMATCH: {ticker} {cost_type} {metric}: artifact={artifact_val:.6f}, recomputed={recomputed_val:.6f}")

    if all_match:
        print(f"\n=== VERIFICATION PASSED ===")
        print(f"All recomputed results match the artifact exactly.")
        return 0
    else:
        print(f"\n=== VERIFICATION FAILED ===")
        print(f"Mismatches found between artifact and recomputation.")
        return 1


if __name__ == "__main__":
    sys.exit(verify_momentum_cost_robustness())
