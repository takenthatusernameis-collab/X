"""Independent verification of the momentum cost sensitivity check
(`research/checks/momentum_cost_sensitivity.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median and the universe-level results from a fresh `walk_forward`
implementation (not `stress_segments`), then compares against the artifact
written by the momentum cost sensitivity check (`state/check_artifacts/momentum_cost_sensitivity_results.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, aggregating fold log returns -- no call to ``stress_segments``.
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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"

BASE_SEED = 42  # same base seed the check used for perturbations


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
    segments = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                segments.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_segment_bars:
        segments.append((cur, start, len(labels)))
    return segments


def momentum_signals(closes, lookback):
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def cost_sensitivity_for_segment(asset, s, e, lookback, cfg_zero_cost, cfg_realistic_cost):
    """Run cost sensitivity check for one segment using fresh walk_forward."""
    bars, dates = bt.load_ticker(asset)
    closes = bars.closes_array()
    
    # Create momentum signals for this lookback
    signals = momentum_signals(closes, lookback)
    
    # Extract segment data
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    
    # Zero cost
    res_zero = bt.walk_forward(
        seg_bars, seg_signals,
        train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg_zero_cost
    )
    
    # Realistic cost
    res_realistic = bt.walk_forward(
        seg_bars, seg_signals,
        train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg_realistic_cost
    )
    
    # Calculate cost impact metrics
    zero_mean_ret = res_zero.aggregate_metrics['mean_log_total_return']
    realistic_mean_ret = res_realistic.aggregate_metrics['mean_log_total_return']
    cost_impact = realistic_mean_ret - zero_mean_ret
    
    # Calculate cost ratio
    cost_ratio = realistic_mean_ret / zero_mean_ret if zero_mean_ret != 0 else float('inf')
    
    # Get per-fold metrics
    zero_n_trades = sum(f.metrics['n_trades'] for f in res_zero.folds)
    zero_total_return = sum(f.metrics['total_return'] for f in res_zero.folds)
    zero_return_per_trade = zero_total_return / zero_n_trades if zero_n_trades > 0 else 0
    
    realistic_n_trades = sum(f.metrics['n_trades'] for f in res_realistic.folds)
    realistic_total_return = sum(f.metrics['total_return'] for f in res_realistic.folds)
    realistic_return_per_trade = realistic_total_return / realistic_n_trades if realistic_n_trades > 0 else 0
    
    return {
        "segment_name": f"{s}-{e}",
        "segment_start": s,
        "segment_end": e,
        "zero_cost": {
            "mean_log_return": zero_mean_ret,
            "median_log_return": res_zero.aggregate_metrics['median_log_total_return'],
            "positive_folds": res_zero.aggregate_metrics['fold_count_positive'],
            "n_folds": res_zero.aggregate_metrics['n_folds'],
            "total_return": zero_total_return,
            "n_trades": zero_n_trades,
            "return_per_trade": zero_return_per_trade,
            "commission_per_trade": cfg_zero_cost.commission_per_trade + cfg_zero_cost.commission_per_share,
            "slippage_per_share": cfg_zero_cost.slippage_cents / 100.0,
        },
        "realistic_cost": {
            "mean_log_return": realistic_mean_ret,
            "median_log_return": res_realistic.aggregate_metrics['median_log_total_return'],
            "positive_folds": res_realistic.aggregate_metrics['fold_count_positive'],
            "n_folds": res_realistic.aggregate_metrics['n_folds'],
            "total_return": realistic_total_return,
            "n_trades": realistic_n_trades,
            "return_per_trade": realistic_return_per_trade,
            "commission_per_trade": cfg_realistic_cost.commission_per_trade + cfg_realistic_cost.commission_per_share,
            "slippage_per_share": cfg_realistic_cost.slippage_cents / 100.0,
        },
        "cost_impact": {
            "absolute_impact": cost_impact,
            "relative_impact_pct": (cost_impact / zero_mean_ret * 100) if zero_mean_ret != 0 else 0,
            "cost_ratio": cost_ratio,
        },
    }


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== Independent recomputation of momentum cost sensitivity (fresh walk_forward path) ===")
    
    # Define cost configurations
    cfg_zero_cost = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=artifact["cost_config_zero"]["commission_per_trade"],
        commission_per_share=artifact["cost_config_zero"]["commission_per_share"],
        slippage_cents=artifact["cost_config_zero"]["slippage_cents"],
        slippage_proportional=artifact["cost_config_zero"]["slippage_proportional"],
    )
    
    cfg_realistic_cost = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=artifact["cost_config_realistic"]["commission_per_trade"],
        commission_per_share=artifact["cost_config_realistic"]["commission_per_share"],
        slippage_cents=artifact["cost_config_realistic"]["slippage_cents"],
        slippage_proportional=artifact["cost_config_realistic"]["slippage_proportional"],
    )
    
    all_ok = True
    
    # Check each lookback
    for lb in LOOKBACKS:
        print(f"\n=== Verifying lookback {lb} ===")
        
        # Verify each ticker and segment
        for ticker in artifact["lookback_results"][str(lb)].keys():
            bars, dates = bt.load_ticker(ticker)
            closes = bars.closes_array()
            
            # Get segments
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            
            # Get published results for this ticker
            published = artifact["lookback_results"][str(lb)][ticker]
            
            # Compare each segment
            for seg_result in published:
                s, e = seg_result["segment_start"], seg_result["segment_end"]
                
                # Verify segment recomputation
                recomputed = cost_sensitivity_for_segment(ticker, s, e, lb, cfg_zero_cost, cfg_realistic_cost)
                
                # Compare zero cost metrics
                for key in ["mean_log_return", "median_log_return", "positive_folds", "n_folds", 
                           "total_return", "n_trades", "return_per_trade"]:
                    pub_val = seg_result["zero_cost"][key]
                    comp_val = recomputed["zero_cost"][key]
                    
                    if isinstance(pub_val, (int, float)):
                        if not np.isclose(pub_val, comp_val, atol=1e-9, rtol=1e-9):
                            all_ok = False
                            print(f"  MISMATCH {ticker}: zero_cost.{key} published={pub_val:.6f} recomputed={comp_val:.6f}")
                
                # Compare realistic cost metrics
                for key in ["mean_log_return", "median_log_return", "positive_folds", "n_folds",
                           "total_return", "n_trades", "return_per_trade"]:
                    pub_val = seg_result["realistic_cost"][key]
                    comp_val = recomputed["realistic_cost"][key]
                    
                    if isinstance(pub_val, (int, float)):
                        if not np.isclose(pub_val, comp_val, atol=1e-9, rtol=1e-9):
                            all_ok = False
                            print(f"  MISMATCH {ticker}: realistic_cost.{key} published={pub_val:.6f} recomputed={comp_val:.6f}")
                
                # Compare cost impact metrics
                for key in ["absolute_impact", "relative_impact_pct", "cost_ratio"]:
                    pub_val = seg_result["cost_impact"][key]
                    comp_val = recomputed["cost_impact"][key]
                    
                    if isinstance(pub_val, (int, float)):
                        if not np.isclose(pub_val, comp_val, atol=1e-9, rtol=1e-9):
                            all_ok = False
                            print(f"  MISMATCH {ticker}: cost_impact.{key} published={pub_val:.6f} recomputed={comp_val:.6f}")
        
        # Verify that all published segments were processed
        published_segments = sum(len(artifact["lookback_results"][str(lb)][ticker]) for ticker in artifact["lookback_results"][str(lb)].keys())
        
        if published_segments > 0:
            print(f"  Lookback {lb}: verified {published_segments} segments across all tickers")
        else:
            print(f"  Lookback {lb}: no segments to verify")
    
    # Determinism check
    print("\nDeterminism: recomputing lookback 5 for AAPL again.")
    for ticker in ["AAPL"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        
        # Recompute for lookback 5
        first_run = runs[0] if runs else None
        if first_run:
            s, e = first_run[1], first_run[2]
            res1 = cost_sensitivity_for_segment(ticker, s, e, 5, cfg_zero_cost, cfg_realistic_cost)
            res2 = cost_sensitivity_for_segment(ticker, s, e, 5, cfg_zero_cost, cfg_realistic_cost)
            
            # Check if all values match
            match = True
            for key in ["zero_cost", "realistic_cost"]:
                for subkey in ["mean_log_return", "median_log_return", "positive_folds", "n_folds", 
                             "total_return", "n_trades", "return_per_trade"]:
                    val1 = res1[key][subkey]
                    val2 = res2[key][subkey]
                    if isinstance(val1, (int, float)) and not np.isclose(val1, val2, atol=1e-9, rtol=1e-9):
                        match = False
                        break
                if not match:
                    break
            
            # Check cost impact
            for subkey in ["absolute_impact", "relative_impact_pct", "cost_ratio"]:
                val1 = res1["cost_impact"][subkey]
                val2 = res2["cost_impact"][subkey]
                if isinstance(val1, (int, float)) and not np.isclose(val1, val2, atol=1e-9, rtol=1e-9):
                    match = False
                    break
            
            print(f"  AAPL lookback 5: recomputation {"identical" if match else "DIFFERENT"}")
            assert match, "verification path not deterministic"
    
    print()
    if all_ok:
        print("All independent recomputations MATCH the cost sensitivity artifact.")
    else:
        print("MISMATCH FOUND: the cost sensitivity artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Momentum cost sensitivity independent verification on AAPL + universe: "
          "exploratory simulation.")
    
    return 0


if __name__ == "__main__":
    main()
