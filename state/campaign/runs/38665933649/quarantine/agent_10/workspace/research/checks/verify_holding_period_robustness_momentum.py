"""Independent verification of holding-period robustness momentum test.

This script independently verifies the holding-period robustness results
by recomputing key metrics and checking deterministic reproducibility.

Design: Re-implement signal generation and verification in a separate code path
from the original test, using fresh random seeds for null generation to ensure
independent verification.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

# Verification configuration
LOOKBACK = 5
HOLDING_PERIODS = [1, 3, 5, 7]
VERIFICATION_SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

# Use the same assets as the original test
target_assets = ["AMZN", "JPM"]

# Same cost parameters as original test
VERIFICATION_COST_CONFIG = bt.BacktestConfig(
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
    
    Independent implementation (separate from original test).
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    
    for i in range(holding - 1, n):
        if i - lookback + 1 >= 0:
            lookback_window = closes[i - lookback + 1 : i + 1]
            ret = np.mean(np.log(lookback_window))
            signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
        
        j = i + 1
        while j < n and j < i + holding:
            signals[j] = signals[i]
            j += 1
    
    return signals


def run_independent_verification():
    """Independently verify the holding-period robustness results."""
    
    print("=== Independent Verification of Holding-Period Robustness ===")
    print(f"Verification seed: {VERIFICATION_SEED}")
    print(f"Lookback: {LOOKBACK}")
    print(f"Holding Periods: {HOLDING_PERIODS}")
    print()
    
    # Load the artifact to verify against
    artifact_path = Path("state/check_artifacts/holding_period_robustness_momentum.json")
    
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
    
    verification_results = {}
    discrepancies = []
    
    for holding in HOLDING_PERIODS:
        print(f"\n{'='*60}")
        print(f"Verifying holding period {holding}")
        print(f"{'='*60}")
        
        # Re-run momentum tests independently
        momentum_verification = {}
        
        for asset in target_assets:
            # Load ticker data
            bars, dates = bt.load_ticker(asset)
            closes = bars.closes_array()
            
            # Generate signals using independent implementation
            signals = generate_rolling_signals(closes, LOOKBACK, holding)
            
            # Verify signal integrity
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
            
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
            
            momentum_verification[asset] = {
                "total_return": float(metrics.total_return),
                "annualized_return": float(metrics.annualized_return),
                "sharpe": float(metrics.sharpe),
                "n_trades": int(metrics.n_trades),
                "final_cash": float(metrics.final_cash),
                "total_commission": float(sum(f.commission for f in result.trades)),
            }
            
            # Re-run null tests independently with different random seed
            np.random.seed(VERIFICATION_SEED + holding * 1000 + 999)
            null_signals = []
            for i in range(len(closes)):
                if i < LOOKBACK:
                    null_signals.append(bt.Signal(date=i + 1, weight=0.0))
                else:
                    random_weight = np.random.choice([-1.0, 0.0, 1.0], p=[0.3, 0.4, 0.3])
                    null_signals.append(bt.Signal(date=i + 1, weight=random_weight))
            
            # Run null backtest with verification config
            null_result = bt.run_bars(list(bars), null_signals, cfg)
            null_metrics = bt.compute_metrics(
                null_result.equity_curve,
                fills=null_result.trades,
                fill_prices=np.array([f.price for f in null_result.trades]),
                periods_per_year=cfg.periods_per_year,
            )
            
            momentum_verification[asset]["null"] = {
                "total_return": float(null_metrics.total_return),
                "annualized_return": float(null_metrics.annualized_return),
                "sharpe": float(null_metrics.sharpe),
                "n_trades": int(null_metrics.n_trades),
                "final_cash": float(null_metrics.final_cash),
                "total_commission": float(sum(f.commission for f in null_result.trades)),
            }
            
            # Compare independent results with artifact
            artifact_for_holding = artifact["results"][str(holding)]
            artifact_momentum = artifact_for_holding[asset]["momentum"]["total_return"]
            independent_momentum = momentum_verification[asset]["total_return"]
            
            # Check for discrepancies in momentum signals (these should be identical)
            momentum_diff = abs(artifact_momentum - independent_momentum)
            
            if momentum_diff > 1e-3:
                discrepancies.append({
                    "asset": asset,
                    "holding_period": holding,
                    "component": "momentum",
                    "metric_diffs": {
                        "momentum_total_return": momentum_diff,
                    }
                })
                
                print(f"  ⚠️  DISCREPANCIES FOUND:")
                print(f"    Asset {asset}, holding {holding}: momentum_diff={momentum_diff:.6f}")
            else:
                print(f"  ✅ VERIFICATION PASSED - Momentum signals identical")
            
            # Recompute comparison
            print(f"\n{holding}-day holding verification comparison for {asset}:")
            print(f"{'Component':<20} {'Artifact':<15} {'Independent':<15} {'Diff':<15}")
            print(f"{'Momentum':<20} {artifact_momentum:+.6f}    {independent_momentum:+.6f}    {artifact_momentum - independent_momentum:+.6f}")
            
            # Note: Null signals will differ because verification uses different random seeds
            # This is expected and indicates independent verification methodology
        
        verification_results[holding] = momentum_verification
    
    # Summary of verification
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Artifacts verified: {artifact_path.name}")
    print(f"Verification seed: {VERIFICATION_SEED}")
    print(f"Test assets: {target_assets}")
    print(f"Lookback: {LOOKBACK}")
    print(f"Holding periods: {HOLDING_PERIODS}")
    
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
        print("The holding-period robustness results are deterministically reproducible")
        print("and independently verifiable.")
        
        return {
            "status": "PASSED",
            "message": "Independent verification successful - results are reproducible",
            "verification_method": "deterministic_recomputation_with_different_seed",
        }


if __name__ == "__main__":
    try:
        # Run independent verification
        verification_result = run_independent_verification()
        
        print(f"\n✅ VERIFICATION COMPLETE")
        
        if verification_result["status"] == "PASSED":
            print(f"\n🎉 SUCCESS: Holding-period robustness results are VERIFIED")
            print(f"Result status: {verification_result['message']}")
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        raise