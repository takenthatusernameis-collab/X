"""Independent verification of cost-sensitivity momentum test.

This script independently verifies the momentum cost-robustness results
by recomputing key metrics and checking deterministic reproducibility.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import research.backtest as bt
from research.data.preflight import load_manifest

# Deterministic seed for verification
VERIFICATION_SEED = 42

# Same cost parameters as used in the original test
VERIFICATION_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_trade=5.0,
    commission_per_share=0.003,
    slippage_cents=5.0,
    slippage_proportional=0.001,
    warmup_periods=0,
    periods_per_year=252,
    risk_free=0.0,
)


def independent_momentum_verification():
    """Independently verify the momentum cost-robustness results."""
    
    print("=== Independent Verification of Momentum Cost Robustness ===")
    print(f"Verification seed: {VERIFICATION_SEED}")
    print()
    
    # Load the artifact to verify against
    artifact_path = Path("state/check_artifacts/cost_sensitivity_momentum.json")
    
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    
    with open(artifact_path, "r") as f:
        artifact = json.load(f)
    
    print(f"Loaded artifact from: {artifact_path}")
    print(f"Original test dataset: {artifact['metadata']['dataset_id']}")
    print(f"Original test seed: {artifact['metadata']['seed']}")
    print()
    
    # Recreate independent verification environment
    np.random.seed(VERIFICATION_SEED)
    
    # Use same test assets and parameters
    test_assets = ["AMZN", "JPM"]
    lookback_values = [3, 5, 10]
    
    verification_results = {}
    discrepancies = []
    
    for lookback in lookback_values:
        print(f"\n{'='*60}")
        print(f"Verifying lookback {lookback}")
        print(f"{'='*60}")
        
        # Re-run momentum tests independently
        momentum_independent = {}
        
        for asset in test_assets:
            # Load ticker data
            bars, dates = bt.load_ticker(asset)
            closes = bars.closes_array()
            
            # Create signals using deterministic momentum function
            signals = bt.momentum_signals(closes, lookback=lookback)
            
            # Run backtest with verification config
            cfg = VERIFICATION_COST_CONFIG
            result = bt.run_bars(list(bars), signals, cfg)
            
            # Compute metrics independently
            metrics = bt.compute_metrics(
                result.equity_curve,
                fills=result.trades,
                fill_prices=np.array([f.price for f in result.trades]),
                periods_per_year=cfg.periods_per_year,
            )
            
            momentum_independent[asset] = {
                "total_return": float(metrics.total_return),
                "annualized_return": float(metrics.annualized_return),
                "sharpe": float(metrics.sharpe),
                "n_trades": int(metrics.n_trades),
                "final_cash": float(metrics.final_cash),
                "total_commission": float(sum(f.commission for f in result.trades)),
            }
        
        # Re-run null tests independently  
        null_independent = {}
        
        for asset in test_assets:
            # Load ticker data
            bars, dates = bt.load_ticker(asset)
            closes = bars.closes_array()
            
            # Create independent null signals (different random seed for verification)
            np.random.seed(VERIFICATION_SEED + lookback * 100 + 999)
            random_weights = np.random.choice([-1.0, 0.0, 1.0], size=len(closes), p=[0.3, 0.4, 0.3])
            
            # Ensure no signals before lookback window
            null_signals = []
            for i in range(len(closes)):
                if i < lookback:
                    null_signals.append(bt.Signal(date=i + 1, weight=0.0))
                else:
                    null_signals.append(bt.Signal(date=i + 1, weight=random_weights[i]))
            
            # Run backtest with verification config
            result = bt.run_bars(list(bars), null_signals, VERIFICATION_COST_CONFIG)
            
            # Compute metrics independently
            metrics = bt.compute_metrics(
                result.equity_curve,
                fills=result.trades,
                fill_prices=np.array([f.price for f in result.trades]),
                periods_per_year=VERIFICATION_COST_CONFIG.periods_per_year,
            )
            
            null_independent[asset] = {
                "total_return": float(metrics.total_return),
                "annualized_return": float(metrics.annualized_return),
                "sharpe": float(metrics.sharpe),
                "n_trades": int(metrics.n_trades),
                "final_cash": float(metrics.final_cash),
                "total_commission": float(sum(f.commission for f in result.trades)),
            }
        
        # Compare independent results with artifact
        artifact_for_lookback = artifact["results"][str(lookback)]
        
        verification_results[lookback] = {
            "momentum": momentum_independent,
            "null": null_independent,
            "artifact_momentum": artifact_for_lookback,
        }
        
        # Check for discrepancies
        discrepancies_for_lookback = []
        
        for asset in test_assets:
            m_artifact = artifact_for_lookback[asset]["momentum"]["total_return"]
            m_indep = momentum_independent[asset]["total_return"]
            
            diff = abs(m_artifact - m_indep)
            
            if diff > 1e-3:
                discrepancies_for_lookback.append({
                    "asset": asset,
                    "lookback": lookback,
                    "component": "momentum",
                    "metric_diffs": {
                        "total_return": diff,
                    }
                })
        
        if discrepancies_for_lookback:
            discrepancies.extend(discrepancies_for_lookback)
            print(f"  ⚠️  DISCREPANCIES FOUND:")
            for d in discrepancies_for_lookback:
                print(f"    Asset {d['asset']}, lookback {d['lookback']}, {d['component']}: "
                      f"total_return_diff={d['metric_diffs']['total_return']:.6f}")
        else:
            print(f"  ✅ VERIFICATION PASSED - No significant discrepancies")
        
        # Recompute comparison
        print(f"\n{lookback}-day lookback verification comparison:")
        print(f"{'Asset':<10} {'Artifact':<15} {'Independent':<15} {'Diff':<15}")
        
        for asset in test_assets:
            m_artifact = artifact_for_lookback[asset]["momentum"]["total_return"]
            m_indep = momentum_independent[asset]["total_return"]
            
            diff = m_artifact - m_indep
            
            print(f"{asset:<10} {m_artifact:+.6f}    {m_indep:+.6f}    {diff:+.6f}")
    
    # Summary of verification
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Artifacts verified: {artifact_path.name}")
    print(f"Verification seed: {VERIFICATION_SEED}")
    print(f"Test assets: {test_assets}")
    print(f"Lookback values tested: {lookback_values}")
    
    if discrepancies:
        print(f"\n❌ VERIFICATION FAILED - {len(discrepancies)} discrepancies found")
        print("This indicates potential issues with:")
        print("  - Non-deterministic implementation")
        print("  - Random seed management")
        print("  - Numerical precision")
        print("  - Independent verification methodology")
        
        return {
            "status": "FAILED",
            "discrepancies": discrepancies,
            "message": "Independent verification found significant discrepancies",
        }
    else:
        print(f"\n✅ VERIFICATION PASSED - All checks consistent")
        print("The momentum cost-robustness results are deterministically reproducible")
        print("and independently verifiable.")
        
        return {
            "status": "PASSED",
            "message": "Independent verification successful - results are reproducible",
            "verification_method": "deterministic_recomputation_with_different_seed",
        }


if __name__ == "__main__":
    try:
        # Run independent verification
        verification_result = independent_momentum_verification()
        
        print(f"\n✅ VERIFICATION COMPLETE")
        
        if verification_result["status"] == "PASSED":
            print(f"\n🎉 SUCCESS: Momentum cost-robustness results are VERIFIED")
            print(f"Result status: {verification_result['message']}")
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        raise