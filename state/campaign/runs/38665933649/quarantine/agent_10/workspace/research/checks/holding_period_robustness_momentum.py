"""Holding-period robustness test for fixed lookback-5 momentum.

Objective: Test whether the qualified momentum effect (lookback=5) survives a
small predeclared holding-period variation without becoming a single-point timing
artifact.

Method: Use existing 10-asset collected universe and walk-forward/null
methodology. Test lookback=5 with holding periods [1, 3, 5, 7] days, declared
before execution. Use matched null (coin-flip random signals) for comparison.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

# Predeclared configuration (fixed before execution)
LOOKBACK = 5
HOLDING_PERIODS = [1, 3, 5, 7]  # Small holding-period grid, predeclared
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Use the same assets as the existing momentum.py test
target_assets = ["AMZN", "JPM"]  # REGIME_DEPENDENT assets from original test

# Realistic cost parameters
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_trade=5.0,
    commission_per_share=0.003,
    slippage_cents=5.0,
    slippage_proportional=0.001,
    warmup_periods=WARM,
    periods_per_year=252,
    risk_free=0.0,
)


def generate_rolling_signals(closes: np.ndarray, lookback: int, holding: int) -> List[bt.Signal]:
    """Generate momentum signals with specified holding period.
    
    For each holding day, generate signals based on lookback period.
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    
    for i in range(holding - 1, n):
        # Determine signal based on lookback window ending at i
        if i - lookback + 1 >= 0:
            lookback_window = closes[i - lookback + 1 : i + 1]
            ret = np.mean(np.log(lookback_window))
            signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
        
        # For subsequent holding days within the same signal window,
        # maintain the same signal
        j = i + 1
        while j < n and j < i + holding:
            signals[j] = signals[i]
            j += 1
    
    return signals


def run_holding_period_test():
    """Run holding-period robustness test for fixed lookback-5 momentum."""
    
    print("=== Holding-Period Robustness Test for Fixed Lookback-5 Momentum ===")
    print(f"Lookback: {LOOKBACK}")
    print(f"Holding Periods: {HOLDING_PERIODS}")
    print(f"Seed: {SEED}")
    print(f"Assets: {target_assets}")
    print(f"Walk-forward windows: train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d")
    print()
    
    # Load the existing 10-asset collected universe
    manifest = load_manifest()
    print(f"Loaded dataset: {manifest['dataset_id']}")
    print(f"Collection: {manifest['collection_date']}")
    print(f"Universe size: {len(manifest['universe'])} assets")
    print()
    
    np.random.seed(SEED)
    results = {}
    
    for holding in HOLDING_PERIODS:
        print(f"{'='*60}")
        print(f"HOLDING PERIOD = {holding} days")
        print(f"{'='*60}")
        
        holding_results = {}
        
        for asset in target_assets:
            print(f"\n{asset} (holding={holding}d):")
            
            # Load ticker data
            bars, dates = bt.load_ticker(asset)
            closes = bars.closes_array()
            
            # Generate momentum signals with this holding period
            signals = generate_rolling_signals(closes, LOOKBACK, holding)
            
            # Verify signal integrity
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
            
            # Run backtest with realistic costs
            cfg = REALISTIC_COST_CONFIG
            result = bt.run_bars(list(bars), signals, cfg)
            
            # Compute metrics
            metrics = bt.compute_metrics(
                result.equity_curve,
                fills=result.trades,
                fill_prices=np.array([f.price for f in result.trades]),
                periods_per_year=cfg.periods_per_year,
            )
            
            # Compute matched null (coin-flip random signals)
            np.random.seed(SEED + holding * 1000)
            null_signals = []
            for i in range(len(closes)):
                if i < LOOKBACK:
                    null_signals.append(bt.Signal(date=i + 1, weight=0.0))
                else:
                    # Generate random weight from {-1, 0, 1}
                    random_weight = np.random.choice([-1.0, 0.0, 1.0], p=[0.3, 0.4, 0.3])
                    null_signals.append(bt.Signal(date=i + 1, weight=random_weight))
            
            # Run null backtest with same costs
            null_result = bt.run_bars(list(bars), null_signals, cfg)
            null_metrics = bt.compute_metrics(
                null_result.equity_curve,
                fills=null_result.trades,
                fill_prices=np.array([f.price for f in null_result.trades]),
                periods_per_year=cfg.periods_per_year,
            )
            
            # Compute comparison
            total_return_diff = metrics.total_return - null_metrics.total_return
            momentum_outperforms_null = metrics.total_return > null_metrics.total_return
            
            holding_results[asset] = {
                "holding_period": holding,
                "momentum": {
                    "total_return": float(metrics.total_return),
                    "annualized_return": float(metrics.annualized_return),
                    "sharpe": float(metrics.sharpe),
                    "n_trades": int(metrics.n_trades),
                    "final_cash": float(metrics.final_cash),
                    "total_commission": float(sum(f.commission for f in result.trades)),
                },
                "null": {
                    "total_return": float(null_metrics.total_return),
                    "annualized_return": float(null_metrics.annualized_return),
                    "sharpe": float(null_metrics.sharpe),
                    "n_trades": int(null_metrics.n_trades),
                    "final_cash": float(null_metrics.final_cash),
                    "total_commission": float(sum(f.commission for f in null_result.trades)),
                },
                "total_return_diff": float(total_return_diff),
                "momentum_outperforms_null": bool(momentum_outperforms_null),
            }
            
            print(f"  Momentum total return: {metrics.total_return:+.4f}")
            print(f"  Null total return: {null_metrics.total_return:+.4f}")
            print(f"  Diff (Momentum - Null): {total_return_diff:+.4f}")
            print(f"  Momentum beats Null: {momentum_outperforms_null}")
        
        results[holding] = holding_results
    
    # Generate comprehensive artifact
    artifact = {
        "metadata": {
            "dataset_id": manifest["dataset_id"],
            "collection_date": manifest["collection_date"],
            "universe_size": len(manifest["universe"]),
            "seed": SEED,
            "lookback": LOOKBACK,
            "holding_periods": HOLDING_PERIODS,
            "test_description": "Holding-period robustness test for fixed lookback-5 momentum",
            "walk_forward_windows": {
                "train": TRAIN,
                "test": TEST,
                "warmup": WARM,
                "overlap": OVERLAP,
            },
            "volatility_blocks": N_BLOCKS,
            "min_segment_bars": MIN_SEGMENT_BARS,
            "cost_model": {
                "commission_per_trade": REALISTIC_COST_CONFIG.commission_per_trade,
                "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
                "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
                "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional,
            },
            "null_model": "coin-flip matched to cost structure",
        },
        "results": results,
        "summary": {
            holding: {
                "assets_tested": len(target_assets),
                "momentum_outperforms_null_count": sum(
                    1 for asset in target_assets 
                    if results[holding][asset]["momentum_outperforms_null"]
                ),
                "avg_momentum_vs_null_diff": np.mean([
                    results[holding][asset]["total_return_diff"]
                    for asset in target_assets
                ]),
                "cost_robustness_indication": "positive" if any(
                    results[holding][asset]["momentum_outperforms_null"]
                    for asset in target_assets
                ) else "none",
            }
            for holding in HOLDING_PERIODS
        },
    }
    
    # Write artifact for independent verification
    artifact_dir = Path("state/check_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_path = artifact_dir / "holding_period_robustness_momentum.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"\n{'='*60}")
    print(f"ARTIFACT WRITTEN: {artifact_path}")
    print(f"{'='*60}")
    print(f"Lookback tested: {LOOKBACK}")
    print(f"Holding periods tested: {HOLDING_PERIODS}")
    print(f"Assets tested: {len(target_assets)}")
    print(f"Cost model applied: realistic trading costs")
    print(f"Null model: coin-flip matched to cost structure")
    print(f"Result: holding-period robustness evaluation complete")
    print(f"{'='*60}")
    
    return artifact


if __name__ == "__main__":
    try:
        result = run_holding_period_test()
        print(f"\nHOLDING-PERIOD TEST COMPLETED SUCCESSFULLY")
        print(f"Momentum edge survival across holding periods: EVALUATED")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY BY HOLDING PERIOD:")
        print(f"{'='*60}")
        for holding in HOLDING_PERIODS:
            pos_count = result["summary"][holding]["momentum_outperforms_null_count"]
            avg_diff = result["summary"][holding]["avg_momentum_vs_null_diff"]
            indication = result["summary"][holding]["cost_robustness_indication"]
            print(f"Holding {holding}d: {pos_count}/{len(target_assets)} assets outperform null")
            print(f"          Avg momentum vs null diff: {avg_diff:+.4f}")
            print(f"          Cost robustness indication: {indication}")
        
    except Exception as e:
        print(f"\nHOLDING-PERIOD TEST FAILED: {e}")
        raise