#!/usr/bin/env python3
"""
Agent 06: Holding Period Robustness Test for Momentum Signals

Test a fixed lookback-5 momentum signal with a predeclared small holding-period set
using the existing walk-forward/null methodology.

Primary Question: Does the qualified momentum effect survive a small predeclared
 holding-period variation without becoming a single-point timing artifact?

Hold-period grid (predeclared): [1, 3, 5, 7] days
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest.regime_stability import momentum_signals

# Predeclared holding-period grid (must be declared before execution)
HOLDING_PERIODS = [1, 3, 5, 7]  # small holding-period set

# Walk-forward parameters (from existing methodology)
TRAIN_WINDOW = 252      # 12 months
test_WINDOW = 84        # 4 months
WARMUP = 0              # no warmup for signals
OVERLAP_WINDOW = 60     # overlap between folds
SEED = 42

# Backtest configuration (minimal costs for clear signals)
CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARMUP,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0,
)


def holding_period_momentum_signals(closes: np.ndarray, lookback: int, hold_days: int):
    """Generate momentum signals with variable holding period.
    
    Long the previous lookback-day return and hold for hold_days:
    - if the last lookback days were up, go long for hold_days
    - if they were down, go short for hold_days
    - neutral before the lookback window
    - daily rebalancing (fresh signal every bar)
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        if ret > 0:
            # Long signal: hold for hold_days
            # Each signal day represents entry, but position persists
            signals[i] = bt.Signal(date=i + 1, weight=1.0)
        else:
            # Short signal: hold for hold_days
            signals[i] = bt.Signal(date=i + 1, weight=-1.0)
    
    return signals


def run_holding_period_test():
    """Run walk-forward tests for each holding period."""
    print("=== Agent 06: Holding Period Robustness Test ===")
    print(f"Lookback: 5 days (fixed)")
    print(f"Holding periods: {HOLDING_PERIODS}")
    print(f"Train window: {TRAIN_WINDOW}d, Test window: {test_WINDOW}d")
    print()
    
    results = {}
    
    for hold_days in HOLDING_PERIODS:
        print(f"--- Holding Period: {hold_days} days ---")
        
        # Load real data
        universe = bt.load_universe()
        
        # Generate signals for each asset with this holding period
        per_asset_results = {}
        
        for ticker, (bars, dates) in universe.items():
            closes = bars.closes_array()
            signals = holding_period_momentum_signals(closes, lookback=5, hold_days=hold_days)
            
            # Run walk-forward validation
            wf_result = bt.walk_forward(
                bars=list(bars),
                signals=signals,
                train_window=TRAIN_WINDOW,
                test_window=test_WINDOW,
                warmup=WARMUP,
                overlap_window=OVERLAP_WINDOW,
                cfg=CONFIG,
            )
            
            per_asset_results[ticker] = {
                'holding_period': hold_days,
                'folds': len(wf_result.folds),
                'positive_folds': wf_result.aggregate_metrics["fold_count_positive"],
                'negative_folds': wf_result.aggregate_metrics["fold_count_negative"],
                'zero_folds': wf_result.aggregate_metrics["fold_count_zero"],
                'median_log_return': wf_result.aggregate_metrics["median_log_total_return"],
                'mean_log_return': wf_result.aggregate_metrics["mean_log_total_return"],
                'total_margin_calls': wf_result.aggregate_metrics["total_margin_calls"],
                'n_folds': wf_result.aggregate_metrics["n_folds"],
                'total_oos_periods': wf_result.aggregate_metrics["total_oos_periods"],
            }
        
        # Store results
        results[hold_days] = per_asset_results
        
        # Print summary
        positive_assets = sum(1 for ticker in universe if per_asset_results[ticker]['positive_folds'] > 0)
        print(f"  Positive OOS folds: {positive_assets}/{len(universe)} assets")
        print(f"  Median log return: {wf_result.aggregate_metrics['median_log_total_return']:+.3f}")
        print(f"  Mean log return: {wf_result.aggregate_metrics['mean_log_total_return']:+.3f}")
        print(f"  Fold distribution: P+/{wf_result.aggregate_metrics['fold_count_negative']:+d}/{wf_result.aggregate_metrics['fold_count_negative']:+d}")
        print()
    
    return results


if __name__ == "__main__":
    results = run_holding_period_test()
    
    # Generate robustness artifact
    print("=== Holding Period Robustness Classification ===")
    
    classification = {}
    for hold_days, asset_results in results.items():
        positive_assets = sum(1 for ticker in asset_results if asset_results[ticker]['positive_folds'] > 0)
        total_assets = len(asset_results)
        
        # Use first asset as reference (consistent ordering)
        reference_ticker = list(asset_results.keys())[0]
        classification[hold_days] = {
            'positive_assets': positive_assets,
            'fraction_positive': positive_assets / total_assets,
            'median_log_return': asset_results[reference_ticker]['median_log_return'],
            'status': 'POSITIVE' if positive_assets >= total_assets * 0.5 else 'NEGATIVE'
        }
        
        print(f"Holding period {hold_days} days:")
        print(f"  Positive assets: {positive_assets}/{total_assets} ({classification[hold_days]['fraction_positive']:.1%})")
        print(f"  Median log return: {classification[hold_days]['median_log_return']:+.3f}")
        print(f"  Status: {classification[hold_days]['status']}")
        print()
    
    # Save results to state/check_artifacts/
    import json
    output_path = Path(__file__).resolve().parent / "state/check_artifacts/holding_period_robustness.json"
    
    with open(output_path, 'w') as f:
        json.dump({
            'dataset_id': 'yf-ohlcv-universe-2009-to-2026-10-03',
            'lookback': 5,
            'holding_periods_tested': HOLDING_PERIODS,
            'classification': classification,
            'results': results,
            'metadata': {
                'seed': SEED,
                'train_window': TRAIN_WINDOW,
                'test_window': test_WINDOW,
                'overlap_window': OVERLAP_WINDOW,
                'universe_size': 10,
                'universe': list(bt.load_universe().keys()),
            }
        }, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print("\n=== Test Complete ===")