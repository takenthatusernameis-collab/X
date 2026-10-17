"""Independent verification of cost sensitivity check for momentum lookback 3/5/10.

This script recomputes the cost sensitivity results from the artifact to verify
determinism and accuracy, matching the verification pattern used elsewhere in the repo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_fixed.json"

def momentum_signals(closes, lookback):
    """Momentum signal: long the previous N-day return, hold 1 day, daily rebalance."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def verify_cost_sensitivity_check():
    """Run independent verification of the cost sensitivity check."""
    print("=== Independent Verification of Cost Sensitivity Check ===")
    
    # Load the artifact
    with open(ARTIFACT_PATH, "r") as f:
        artifact = json.load(f)
    
    print(f"  Loaded artifact from {ARTIFACT_PATH}")
    print(f"  Dataset: {artifact['dataset_id']}")
    print(f"  Lookbacks: {artifact['lookbacks']}")
    print(f"  Cost models: {artifact['cost_models']}")
    
    # Run verification for each lookback and cost model
    all_verified = True
    verification_results = []
    
    for lookback in artifact["lookbacks"]:
        print(f"\n  Verifying lookback {lookback}:")
        
        for cost_model in artifact["cost_models"]:
            print(f"    Cost model: {cost_model}")
            
            # Recreate the tickers from manifest
            manifest = load_manifest()
            tickers = {}
            for entry in manifest["entries"]:
                tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
            
            # Set up cost configuration
            if cost_model == "zero":
                cfg = bt.BacktestConfig(
                    initial_capital=1e6,
                    warmup_periods=60,
                    commission_per_trade=0.0,
                    commission_per_share=0.0,
                    slippage_cents=0.0,
                    slippage_proportional=0.0,
                )
            else:  # realistic
                cfg = bt.BacktestConfig(
                    initial_capital=1e6,
                    warmup_periods=60,
                    commission_per_trade=2.0,
                    commission_per_share=0.003,
                    slippage_cents=2.0,
                    slippage_proportional=0.0005,
                )
            
            # Run verification computation
            results = {}
            for asset in tickers:
                bars = tickers[asset]
                closes = bars.closes_array()
                labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
                seg_fn = bt.segment_fn_from_labels(labels)
                
                sig = momentum_signals(closes, lookback)
                res = bt.stress_segments(
                    signals_fn=lambda c, **kw: momentum_signals(c, lookback),
                    bars=list(bars),
                    signals=sig,
                    param_grid=[{"lookback": lookback}],
                    baseline=(("lookback", lookback),),
                    segment_fn=seg_fn,
                    train_window=252,
                    test_window=84,
                    warmup=60,
                    overlap_window=60,
                    periods_per_year=252,
                    min_segment_bars=400,
                    cfg=cfg,
                )
                results[asset] = {
                    "segments": [s.name for s in res.scenarios],
                    "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                    "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                    "candidate_dispersion": round(res.candidate_dispersion, 3),
                    "null_dispersion": round(res.null_dispersion, 3),
                    "verdict": res.overall_verdict,
                    "n_folds": res.n_folds,
                }
            
            # Compare with artifact
            artifact_results = artifact["results"][lookback][f"{cost_model}_cost"]
            match = True
            discrepancies = []
            
            for asset in tickers:
                if asset not in artifact_results:
                    discrepancies.append(f"  Asset {asset}: missing in artifact")
                    match = False
                    continue
                
                art_result = artifact_results[asset]
                ver_result = results[asset]
                
                if art_result["verdict"] != ver_result["verdict"]:
                    discrepancies.append(f"  Asset {asset}: verdict mismatch - artifact: {art_result['verdict']}, verification: {ver_result['verdict']}")
                    match = False
                
                if art_result["medians"] != ver_result["medians"]:
                    discrepancies.append(f"  Asset {asset}: medians mismatch - artifact: {art_result['medians']}, verification: {ver_result['medians']}")
                    match = False
                
                if art_result["null_medians"] != ver_result["null_medians"]:
                    discrepancies.append(f"  Asset {asset}: null_medians mismatch - artifact: {art_result['null_medians']}, verification: {ver_result['null_medians']}")
                    match = False
            
            if match:
                print(f"    [PASS] {cost_model}_cost results match artifact")
                verification_results.append({
                    "lookback": lookback,
                    "cost_model": cost_model,
                    "status": "MATCH",
                    "discrepancies": []
                })
            else:
                print(f"    [FAIL] {cost_model}_cost results do NOT match artifact")
                for discrepancy in discrepancies:
                    print(discrepancy)
                all_verified = False
                verification_results.append({
                    "lookback": lookback,
                    "cost_model": cost_model,
                    "status": "MISMATCH",
                    "discrepancies": discrepancies
                })
    
    print(f"\n=== Verification Summary ===")
    print(f"  Total checks: {len(verification_results)}")
    print(f"  Successful matches: {sum(1 for r in verification_results if r['status'] == 'MATCH')}")
    print(f"  Failed matches: {sum(1 for r in verification_results if r['status'] == 'MISMATCH')}")
    
    if all_verified:
        print("\n✅ ALL VERIFICATIONS PASSED - Cost sensitivity check is deterministic and accurate")
    else:
        print("\n❌ SOME VERIFICATIONS FAILED - Issues found in cost sensitivity check")
    
    return all_verified


if __name__ == "__main__":
    success = verify_cost_sensitivity_check()
    sys.exit(0 if success else 1)
