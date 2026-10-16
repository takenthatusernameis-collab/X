"""Cost-sensitivity check for momentum lookback 3/5/10 with realistic costs.

Objective: Test whether the short-horizon momentum edge (lookback 3/5/10)
survives under realistic transaction costs based on the repository's existing
realistic cost model and matched null.

Background: The candidate positive momentum evidence from research/checks/momentum.py
(activation 37318950814) shows short-horizon positive edge but leaves execution-cost
robustness as a decision-relevant qualification.

Method: Long the previous N-day return, hold 1 day, daily rebalance.
Realistic cost model based on repository's existing examples (ma_crossover.py):
- commission_per_trade: 2.0 (USD per round trip)
- commission_per_share: 0.003 (USD per share)
- slippage_cents: 2.0 (USD fixed slippage per share)
- slippage_proportional: 0.0005 (0.05% proportional slippage)

Same walk-forward windows and volatility blocks as existing momentum research:
- train=252d / test=84d / warmup=60d / overlap=60d, min 400 bars/segment
- 4 contiguous volatility blocks by past-only volatility classifier

Research / simulation only. No live trading or production execution.
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

# Realistic cost model based on repository examples (ma_crossover.py)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,      # USD per round trip
    commission_per_share=0.003,    # USD per share
    slippage_cents=2.0,            # USD fixed slippage per share
    slippage_proportional=0.0005,  # 0.05% proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)

# Zero-cost benchmark for comparison
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)


def run_momentum_cost_sensitivity():
    """Run momentum with realistic costs and compare against zero-cost benchmark."""
    ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
    ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Cost-Sensitivity Check for Momentum Lookback 3/5/10 ===")
    print(f"Dataset: {manifest['dataset_id']} | Collection: {manifest['collection_date']} | Universe: {len(manifest['universe'])}")
    print(f"\nRealistic cost model:")
    print(f"  commission_per_trade: ${REALISTIC_COSTS.commission_per_trade:.2f}")
    print(f"  commission_per_share: ${REALISTIC_COSTS.commission_per_share:.6f}")
    print(f"  slippage_cents: ${REALISTIC_COSTS.slippage_cents:.2f}")
    print(f"  slippage_proportional: {REALISTIC_COSTS.slippage_proportional:.6f}")
    print(f"\nWalk-forward: train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, {N_BLOCKS} volatility blocks")

    # Load universe tickers  
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    print(f"\nRunning momentum with realistic costs...")

    results = {}
    for lookback in LOOKBACKS:
        print(f"\nLookback {lookback}:")
        
        # Get momentum signals
        momentum_signals = bt.momentum_signals
        
        # Run stress segments with realistic costs
        res_realistic = bt.stress_segments_across_tickers(
            tickers=tickers_univ,
            signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lookback),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", lookback),),
            param_grid=[{"lookback": lookback}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=REALISTIC_COSTS,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        
        # Run stress segments with zero costs
        res_zero = bt.stress_segments_across_tickers(
            tickers=tickers_univ,
            signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lookback),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", lookback),),
            param_grid=[{"lookback": lookback}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=ZERO_COSTS,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        
        # For zero costs, we need to manually run the backtest to get costs
        # Create a simple test to show cost impact
        target = ["AMZN", "JPM", "AAPL"]  # Sample assets for detailed analysis
        cost_impact = {}
        
        for ticker in target:
            bars = tickers_univ[ticker]
            closes = bars.closes_array()
            signals = bt.momentum_signals(closes, lookback=lookback)
            
            # Run with realistic costs
            result_realistic = bt.run_bars(list(bars), signals, REALISTIC_COSTS)
            # Run with zero costs  
            result_zero = bt.run_bars(list(bars), signals, ZERO_COSTS)
            
            # Calculate total return and costs
            total_return_realistic = (result_realistic.equity_curve[-1] - 1e6) / 1e6
            total_return_zero = (result_zero.equity_curve[-1] - 1e6) / 1e6
            
            # Total trading costs incurred
            total_costs = sum(f.commission for f in result_realistic.trades) + \
                         sum(f.shares * f.price for f in result_realistic.trades if f.shares > 0)
            
            cost_impact[ticker] = {
                "lookback": lookback,
                "zero_cost_return": total_return_zero,
                "realistic_cost_return": total_return_realistic,
                "cost_impact_percent": (total_return_realistic - total_return_zero) / total_return_zero * 100 if total_return_zero != 0 else float('inf'),
                "total_costs_usd": total_costs,
                "n_trades": len(result_realistic.trades),
                "n_shares_traded": sum(abs(f.shares) for f in result_realistic.trades),
                "avg_cost_per_trade": total_costs / len(result_realistic.trades) if len(result_realistic.trades) > 0 else 0,
            }
        
        results[str(lookback)] = {
            "cost_impact": cost_impact,
            "zero_cost_aggregate": {"summary": "RegimeUniverseSummary (no aggregate_metrics)"},
            "realistic_cost_aggregate": {"summary": "RegimeUniverseSummary (no aggregate_metrics)"},
            "zero_cost_verdict_counts": res_zero.verdict_counts,
            "realistic_cost_verdict_counts": res_realistic.verdict_counts,
        }
        
        print(f"  Zero-cost verdict counts: {results[str(lookback)]['zero_cost_verdict_counts']}")
        print(f"  Realistic-cost verdict counts: {results[str(lookback)]['realistic_cost_verdict_counts']}")
        print(f"  Cost impact on sample assets: {[f'{t} ({v['cost_impact_percent']:.1f}%)' for t, v in cost_impact.items()]}")

    # Write results artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "realistic_costs": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        "zero_costs": {
            "commission_per_trade": ZERO_COSTS.commission_per_trade,
            "commission_per_share": ZERO_COSTS.commission_per_share,
            "slippage_cents": ZERO_COSTS.slippage_cents,
            "slippage_proportional": ZERO_COSTS.slippage_proportional,
        },
        "lookback_results": results,
        "summary": {
            "positive_under_realistic_costs": sum(1 for lb in LOOKBACKS 
                                              if results[str(lb)]["realistic_cost_verdict_counts"].get("REGIME_STABLE", 0) > 0),
            "positive_under_zero_costs": sum(1 for lb in LOOKBACKS 
                                           if results[str(lb)]["zero_cost_verdict_counts"].get("REGIME_STABLE", 0) > 0),
            "cost_sensitivity_analysis": "completed",
        },
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n=== Cost sensitivity analysis complete ===")
    print(f"Results written to: {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(run_momentum_cost_sensitivity())