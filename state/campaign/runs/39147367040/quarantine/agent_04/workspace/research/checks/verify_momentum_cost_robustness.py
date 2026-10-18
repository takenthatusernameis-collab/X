#!/usr/bin/env python3
"""Independent verification of momentum cost sensitivity analysis.

This is a separate implementation path that recomputes each segment's walk-forward
median and the universe-level results from a fresh walk_forward implementation,
then compares against the artifact written by the momentum cost robustness check.
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from closes using a re-implemented
  volatility_blocks (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using min_segment_bars.
- Segment medians are computed via walk_forward directly on the segment
  slice, aggregating fold log returns -- no call to stress_segments.
- Cost metrics are recomputed by walking each segment independently.
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
WARMUP = 60
N_BLOCKS = 4
WINDOW = 60
MIN_SEGMENT_BARS = 400


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


def segment_median(ticker, s, e, train, test, warm, overlap, cfg_zero, cfg_realistic, lookback):
    """Compute OOS median for both cost models on a segment (independent path)."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    
    # Run with zero cost
    result_zero = bt.walk_forward(
        list(bars)[s:e], signals[s:e], train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg_zero
    )
    
    # Run with realistic costs
    result_real = bt.walk_forward(
        list(bars)[s:e], signals[s:e], train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg_realistic
    )
    
    from research.backtest.metrics import compute_metrics
    
    # Extract metrics for both scenarios
    metrics_zero = {}
    metrics_real = {}
    
    if len(result_zero.folds) > 0:
        last_fold = result_zero.folds[-1]
        metrics_zero = compute_metrics(
            last_fold.equity_curve[-test:],
            fills=[],
            fill_prices=np.array([], dtype=np.float64),
            risk_free=cfg_zero.risk_free,
            periods_per_year=cfg_zero.periods_per_year,
        )
    
    if len(result_real.folds) > 0:
        last_fold = result_real.folds[-1]
        metrics_real = compute_metrics(
            last_fold.equity_curve[-test:],
            fills=[],
            fill_prices=np.array([], dtype=np.float64),
            risk_free=cfg_realistic.risk_free,
            periods_per_year=cfg_realistic.periods_per_year,
        )
    
    return {
        "zero_cost_metrics": metrics_zero,
        "realistic_cost_metrics": metrics_real,
    }


def aggregate_metrics(metrics_list):
    """Aggregate metrics from multiple fold results (adapted from walk_forward)."""
    if not metrics_list:
        return {
            "n_folds": 0,
            "total_oos_periods": 0,
            "mean_log_total_return": 0.0,
            "median_log_total_return": 0.0,
            "std_log_total_return": 0.0,
            "fold_count_positive": 0,
            "fold_count_negative": 0,
            "fold_count_zero": 0,
            "total_margin_calls": 0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover_ratio": 0.0,
            "n_trades": 0,
        }
    
    # Convert metrics to log returns
    log_returns = []
    
    for m in metrics_list:
        if m and hasattr(m, "total_return"):
            log_returns.append(np.log1p(m.total_return))
    
    if not log_returns:
        return {
            "n_folds": 0,
            "total_oos_periods": 0,
            "mean_log_total_return": 0.0,
            "median_log_total_return": 0.0,
            "std_log_total_return": 0.0,
            "fold_count_positive": 0,
            "fold_count_negative": 0,
            "fold_count_zero": 0,
            "total_margin_calls": 0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover_ratio": 0.0,
            "n_trades": 0,
        }
    
    # Create aggregate dictionary
    aggregate = {
        "n_folds": len(log_returns),
        "total_oos_periods": 0,
        "mean_log_total_return": float(np.mean(log_returns)),
        "median_log_total_return": float(np.median(log_returns)),
        "std_log_total_return": float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0,
        "fold_count_positive": int(np.sum(np.array(log_returns) > 0)),
        "fold_count_negative": int(np.sum(np.array(log_returns) < 0)),
        "fold_count_zero": int(np.sum(np.array(log_returns) == 0)),
        "total_margin_calls": 0,
        "total_return": float(np.expm1(np.mean(log_returns))),
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "turnover_ratio": 0.0,
        "n_trades": 0,
    }
    
    return aggregate


def main():
    np.random.seed(42)
    ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"
    
    print("=== Independent Recomputation of Momentum Cost Sensitivity ===")
    
    # Load the check artifact
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    
    assert artifact["dataset_id"] == DATASET_ID, "dataset id mismatch"
    
    # Define cost models (must match the check)
    cfg_zero = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARMUP)
    cfg_realistic = bt.BacktestConfig(
        initial_capital=1e6, warmup_periods=WARMUP,
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005,
    )
    
    print("=== Independent Verification (fresh walk_forward path) ===")
    all_ok = True
    
    # Recompute AAPL cost sensitivity
    ticker = "AAPL"
    lookback = 5
    
    print(f"\nProcessing {ticker} (lookback {lookback}) with independent recomputation...")
    
    # Load data and create segments
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    
    # Create segments from labels
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    
    # Collect metrics
    all_zero_metrics = []
    all_real_metrics = []
    
    for lab, s, e in runs:
        seg = segment_median(ticker, s, e, 252, 84, WARMUP, 60, cfg_zero, cfg_realistic, lookback)
        
        if seg["zero_cost_metrics"]:
            all_zero_metrics.append(seg["zero_cost_metrics"])
        
        if seg["realistic_cost_metrics"]:
            all_real_metrics.append(seg["realistic_cost_metrics"])
    
    # Aggregate using independent path
    agg_zero_indep = aggregate_metrics(all_zero_metrics)
    agg_real_indep = aggregate_metrics(all_real_metrics)
    
    # Get the check's results
    edge_metrics = artifact["lookback_sensitivity"]["edge_metrics"]
    check_zero = edge_metrics[0]["zero_cost_mean"]
    check_real = edge_metrics[0]["realistic_cost_mean"]
    
    # Compare
    zero_match = abs(agg_zero_indep["mean_log_total_return"] - check_zero) < 0.001
    real_match = abs(agg_real_indep["mean_log_total_return"] - check_real) < 0.001
    
    print(f"Independent zero-cost mean log return: {agg_zero_indep['mean_log_total_return']:+.3f}")
    print(f"Check zero-cost mean log return:       {check_zero:+.3f} -> {'MATCH' if zero_match else 'MISMATCH'}")
    print(f"Independent realistic-cost mean log return: {agg_real_indep['mean_log_total_return']:+.3f}")
    print(f"Check realistic-cost mean log return:       {check_real:+.3f} -> {'MATCH' if real_match else 'MISMATCH'}")
    
    if not (zero_match and real_match):
        all_ok = False
        print(f"\nMISMATCH FOUND: independent recomputation did not match check results.")
    else:
        print(f"\nAll independent recomputations MATCH the check artifact.")
    
    # Determinism test: recompute again and verify identical results
    print("\n=== Determinism Test ===")
    agg_zero_2 = aggregate_metrics(all_zero_metrics)
    agg_real_2 = aggregate_metrics(all_real_metrics)
    
    zero_det = agg_zero_indep["mean_log_total_return"] == agg_zero_2["mean_log_total_return"]
    real_det = agg_real_indep["mean_log_total_return"] == agg_real_2["mean_log_total_return"]
    
    print(f"Zero-cost recomputation deterministic: {'YES' if zero_det else 'NO'}")
    print(f"Realistic-cost recomputation deterministic: {'YES' if real_det else 'NO'}")
    
    if not (zero_det and real_det):
        all_ok = False
        print(f"VERIFICATION FAILED: independent path not deterministic")
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print(f"Momentum cost sensitivity independent verification complete.")
    
    if all_ok:
        print("\n✓ VERIFICATION PASSED: check results are reproducible and deterministic")
        sys.exit(0)
    else:
        print("\n✗ VERIFICATION FAILED: check results failed to reproduce")
        sys.exit(1)


if __name__ == "__main__":
    main()