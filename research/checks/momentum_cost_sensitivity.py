"""Cost sensitivity check for momentum with lookback 3/5/10.

Objective: test whether the short-horizon momentum edge observed at lookback=5
survives realistic transaction costs (zero vs realistic cost models) on the
collected real-data universe.

Method: long the previous lookback-day return; hold 1 day; daily rebalance;
lookbacks 3, 5, 10; same walk-forward settings and volatility blocks as
`momentum.py`. Compare zero cost vs realistic cost models.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

def run_cost_sensitivity_check(ticker, bars, lookback, cfg_zero_cost, cfg_realistic_cost):
    """Run cost sensitivity check for one ticker and lookback using walk_forward."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    
    # Get segments from labels
    segments = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= MIN_SEGMENT_BARS:
                segments.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= MIN_SEGMENT_BARS:
        segments.append((cur, start, len(labels)))
    
    results = []
    for segment_name, seg_start, seg_end in segments:
        seg_bars = list(bars)[seg_start:seg_end]
        seg_closes = closes[seg_start:seg_end]
        
        # Create momentum signals for this lookback
        signals = bt.momentum_signals(seg_closes, lookback=lookback)
        
        # Zero cost
        res_zero = bt.walk_forward(
            seg_bars, signals,
            train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=cfg_zero_cost
        )
        
        # Realistic cost
        res_realistic = bt.walk_forward(
            seg_bars, signals,
            train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=cfg_realistic_cost
        )
        
        # Calculate cost impact metrics (using aggregate_metrics keys)
        zero_mean_ret = res_zero.aggregate_metrics['mean_log_total_return']
        realistic_mean_ret = res_realistic.aggregate_metrics['mean_log_total_return']
        cost_impact = realistic_mean_ret - zero_mean_ret  # negative = cost hurts performance
        
        # Calculate cost ratio (how much realistic cost reduces returns)
        cost_ratio = realistic_mean_ret / zero_mean_ret if zero_mean_ret != 0 else float('inf')
        
        # Get the actual commission and slippage per trade from configs
        zero_commission = cfg_zero_cost.commission_per_trade + cfg_zero_cost.commission_per_share
        realistic_commission = cfg_realistic_cost.commission_per_trade + cfg_realistic_cost.commission_per_share
        
        # Estimate average slippage per share
        zero_slippage = cfg_zero_cost.slippage_cents / 100.0
        realistic_slippage = cfg_realistic_cost.slippage_cents / 100.0
        
        # Note: n_trades and total_return are per-fold metrics, not in aggregate_metrics
        # We can get total metrics by looking at all folds
        zero_n_trades = sum(f.metrics['n_trades'] for f in res_zero.folds)
        zero_total_return = sum(f.metrics['total_return'] for f in res_zero.folds)
        zero_return_per_trade = zero_total_return / zero_n_trades if zero_n_trades > 0 else 0
        
        realistic_n_trades = sum(f.metrics['n_trades'] for f in res_realistic.folds)
        realistic_total_return = sum(f.metrics['total_return'] for f in res_realistic.folds)
        realistic_return_per_trade = realistic_total_return / realistic_n_trades if realistic_n_trades > 0 else 0
        
        results.append({
            "segment_name": segment_name,
            "segment_start": seg_start,
            "segment_end": seg_end,
            "zero_cost": {
                "mean_log_return": zero_mean_ret,
                "median_log_return": res_zero.aggregate_metrics['median_log_total_return'],
                "positive_folds": res_zero.aggregate_metrics['fold_count_positive'],
                "n_folds": res_zero.aggregate_metrics['n_folds'],
                "total_return": zero_total_return,
                "n_trades": zero_n_trades,
                "return_per_trade": zero_return_per_trade,
                "commission_per_trade": zero_commission,
                "slippage_per_share": zero_slippage,
            },
            "realistic_cost": {
                "mean_log_return": realistic_mean_ret,
                "median_log_return": res_realistic.aggregate_metrics['median_log_total_return'],
                "positive_folds": res_realistic.aggregate_metrics['fold_count_positive'],
                "n_folds": res_realistic.aggregate_metrics['n_folds'],
                "total_return": realistic_total_return,
                "n_trades": realistic_n_trades,
                "return_per_trade": realistic_return_per_trade,
                "commission_per_trade": realistic_commission,
                "slippage_per_share": realistic_slippage,
            },
            "cost_impact": {
                "absolute_impact": cost_impact,
                "relative_impact_pct": (cost_impact / zero_mean_ret * 100) if zero_mean_ret != 0 else 0,
                "cost_ratio": cost_ratio,
            },
        })
    
    return results

def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== Cost sensitivity check for momentum lookback 3 / 5 / 10 ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Define cost configurations (consistent with repository examples)
    cfg_zero_cost = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    )
    
    cfg_realistic_cost = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )

    # Load all tickers from the manifest
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    print(f"Loaded {len(tickers)} tickers from {DATASET_ID}")

    # Run cost sensitivity test for each lookback
    lookback_results = {}
    for lb in LOOKBACKS:
        print(f"\n=== Lookback {lb} ===")
        lookback_results[str(lb)] = {}
        
        for ticker in tickers.keys():
            bars = tickers[ticker]
            results = run_cost_sensitivity_check(ticker, bars, lb, cfg_zero_cost, cfg_realistic_cost)
            lookback_results[str(lb)][ticker] = results
            print(f"  {ticker}: processed {len(results)} segments")
            
            # Print summary for this ticker and lookback
            if results:
                zero_costs = [r["zero_cost"]["mean_log_return"] for r in results if r["segment_name"]]
                realistic_costs = [r["realistic_cost"]["mean_log_return"] for r in results if r["segment_name"]]
                if zero_costs and realistic_costs:
                    zero_mean = np.mean(zero_costs)
                    realistic_mean = np.mean(realistic_costs)
                    cost_impact = realistic_mean - zero_mean
                    cost_impact_pct = (cost_impact / zero_mean * 100) if zero_mean != 0 else 0
                    print(f"    Zero cost: mean log ret = {zero_mean:+.3f}")
                    print(f"    Realistic cost: mean log ret = {realistic_mean:+.3f}")
                    print(f"    Cost impact: {cost_impact:+.3f} ({cost_impact_pct:+.1f}%)")

    # Create artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        lookback_results=lookback_results,
        cost_config_zero={
            "commission_per_trade": cfg_zero_cost.commission_per_trade,
            "commission_per_share": cfg_zero_cost.commission_per_share,
            "slippage_cents": cfg_zero_cost.slippage_cents,
            "slippage_proportional": cfg_zero_cost.slippage_proportional,
        },
        cost_config_realistic={
            "commission_per_trade": cfg_realistic_cost.commission_per_trade,
            "commission_per_share": cfg_realistic_cost.commission_per_share,
            "slippage_cents": cfg_realistic_cost.slippage_cents,
            "slippage_proportional": cfg_realistic_cost.slippage_proportional,
        },
        note="cost sensitivity implementation using walk_forward with realistic BacktestConfig",
    )
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== Cost sensitivity artifact written to: {ARTIFACT_PATH} ===")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("      This test examines whether momentum edges survive realistic transaction costs.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())