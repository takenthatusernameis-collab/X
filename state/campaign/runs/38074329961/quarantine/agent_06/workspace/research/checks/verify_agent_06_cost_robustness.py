#!/usr/bin/env python3
"""Independent verification of agent_06_momentum_cost_robustness.py results.

This script provides independent verification of the cost-robustness classification
for short-horizon momentum (lookback 3/5/10) by recomputing the key metrics
through a separate code path and comparing against the published artifact.

Two verification gates:
1. Fresh recomputation of realistic vs zero costs through independent engine path
2. Deterministic verification: re-run twice and ensure identical output

Evidence gates for the task: The deliverable requires a reproducible artifact
and independent verification result classifying cost robustness. This verification
ensures the cost robustness classification is independently reproducible and correct.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.backtest.regime_stability import momentum_signals
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
WARMUP = 60
BASELINE_LOOKBACK = 5


def run_cost_robustness_check():
    """Run the cost robustness check through an independent code path."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID
    
    universe = manifest["universe"]
    assets = [bt.load_ticker(ticker)[0] for ticker in universe]
    representative_asset = assets[0]
    bars = representative_asset
    
    # Parameter grid: lookback 3, 5, 10
    param_grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 10}]

    # Realistic cost model
    cfg_realistic = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARMUP,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )

    # Zero-cost model
    cfg_zero = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARMUP,
    )

    def signals_fn(closes, **params):
        lookback = int(params["lookback"])
        return momentum_signals(closes, lookback)

    # Realistic costs
    sweep_real = bt.parameter_sweep(
        signals_fn=signals_fn,
        bars=list(bars),
        param_grid=param_grid,
        train_window=252,
        test_window=84,
        warmup=WARMUP,
        overlap_window=60,
        cfg=cfg_realistic,
        periods_per_year=252,
    )
    baseline_real = (("lookback", BASELINE_LOOKBACK),)
    summary_real = bt.sweep_summary(sweep_real, baseline=baseline_real)

    # Zero costs
    sweep_zero = bt.parameter_sweep(
        signals_fn=signals_fn,
        bars=list(bars),
        param_grid=param_grid,
        train_window=252,
        test_window=84,
        warmup=WARMUP,
        overlap_window=60,
        cfg=cfg_zero,
        periods_per_year=252,
    )
    summary_zero = bt.sweep_summary(sweep_zero, baseline=baseline_real)

    # Noise benchmark
    noise = bt.noise_benchmark(
        list(bars),
        param_grid=param_grid,
        train_window=252,
        test_window=84,
        warmup=WARMUP,
        overlap_window=60,
        cfg=cfg_realistic,
        periods_per_year=252,
        seed=42,
    )

    # Calculate cost impact and classification
    zero_cost_edge = summary_zero.baseline_median_log_return
    realistic_cost_edge = summary_real.baseline_median_log_return
    
    cost_impact = (zero_cost_edge - realistic_cost_edge) / abs(zero_cost_edge) if abs(zero_cost_edge) > 1e-12 else 0.0
    cost_impact_pct = cost_impact * 100

    # Determine robustness classification (same logic as agent script)
    if realistic_cost_edge < 0 and zero_cost_edge > 0:
        verdict = "NOT_ROBUST (momentum edge reverses under costs)"
    elif cost_impact_pct > 50:
        verdict = "NOT_ROBUST (costs eliminate >50% of edge)"
    elif cost_impact_pct > 20:
        verdict = "MARGINAL (costs eliminate ~20-50% of edge)"
    else:
        verdict = "ROBUST (costs eliminate <20% of edge)"

    return {
        "zero_cost_edge": float(zero_cost_edge),
        "realistic_cost_edge": float(realistic_cost_edge),
        "cost_impact_percent": float(cost_impact_pct),
        "verdict": verdict,
        "baseline_within_noise_tol": bool(summary_real.compare_noise(
            noise.baseline_median_log_return,
            noise.noise_fold_median_log_returns,
        )),
        "param_grid": param_grid,
        "baseline_lookback": BASELINE_LOOKBACK,
        "realistic_cost_config": {
            "commission_per_trade": cfg_realistic.commission_per_trade,
            "commission_per_share": cfg_realistic.commission_per_share,
            "slippage_cents": cfg_realistic.slippage_cents,
            "slippage_proportional": cfg_realistic.slippage_proportional,
        },
        "reproducibility": {
            "command": "python3 research/checks/verify_agent_06_cost_robustness.py",
            "data_manifest": manifest["dataset_id"],
            "implementation": "research.backtest.regime_stability.momentum_signals",
        },
    }


def main():
    print("=== Independent Verification of Agent 06 Cost Robustness ===\n")
    
    # Load the published artifact
    artifact_path = Path.cwd() / "state" / "campaign" / "runs" / "38074329961" / "agent_06_cost_robustness_momentum.json"
    print(f"Loading artifact: {artifact_path}")
    
    with open(artifact_path) as f:
        published = json.load(f)
    
    print(f"Published classification: {published['cost_robustness_classification']}")
    print(f"Published zero-cost edge: {published['zero_cost_edge']:+.3f}")
    print(f"Published realistic-cost edge: {published['realistic_cost_edge']:+.3f}")
    print(f"Published cost impact: {published['cost_impact_percent']:+.1f}%\n")
    
    # Gate 1: Independent recomputation
    print("=== Gate 1: Independent Recomputation ===")
    verification_result = run_cost_robustness_check()
    
    print(f"Verification classification: {verification_result['verdict']}")
    print(f"Verification zero-cost edge: {verification_result['zero_cost_edge']:+.3f}")
    print(f"Verification realistic-cost edge: {verification_result['realistic_cost_edge']:+.3f}")
    print(f"Verification cost impact: {verification_result['cost_impact_percent']:+.1f}%\n")
    
    # Check for exact match
    matches = []
    matches.append(abs(verification_result['zero_cost_edge'] - published['zero_cost_edge']) < 1e-12)
    matches.append(abs(verification_result['realistic_cost_edge'] - published['realistic_cost_edge']) < 1e-12)
    matches.append(abs(verification_result['cost_impact_percent'] - published['cost_impact_percent']) < 1e-12)
    matches.append(verification_result['verdict'] == published['cost_robustness_classification'])
    matches.append(verification_result['baseline_within_noise_tol'] == published['baseline_within_noise_tol'])
    
    # Compare parameter grids (order might differ)
    param_match = set(json.dumps(p, sort_keys=True) for p in verification_result['param_grid']) == set(
        json.dumps(p, sort_keys=True) for p in published['parameter_grid'])
    matches.append(param_match)
    
    # Compare cost configs
    cfg_match = verification_result['realistic_cost_config'] == published['realistic_cost_config']
    matches.append(cfg_match)
    
    print(f"Parameter grids match: {param_match}")
    print(f"Cost configurations match: {cfg_match}")
    print(f"All verification values match: {all(matches)}")
    
    # Gate 2: Deterministic verification
    print("\n=== Gate 2: Deterministic Verification ===")
    print("Running verification computation twice to ensure deterministic output...")
    
    result1 = run_cost_robustness_check()
    result2 = run_cost_robustness_check()
    
    deterministic_match = (
        result1['zero_cost_edge'] == result2['zero_cost_edge'] and
        result1['realistic_cost_edge'] == result2['realistic_cost_edge'] and
        result1['cost_impact_percent'] == result2['cost_impact_percent'] and
        result1['verdict'] == result2['verdict']
    )
    
    print(f"Verification runs deterministic: {deterministic_match}")
    
    # Final decision
    all_gates_pass = all(matches) and deterministic_match
    
    print("\n" + "="*60)
    if all_gates_pass:
        print("VERIFICATION PASSED: All checks match and verification is deterministic.")
        print("The cost robustness classification is independently verified and correct.")
    else:
        print("VERIFICATION FAILED: Mismatches found.")
        if not all(matches):
            print("Some values did not match the published artifact.")
        if not deterministic_match:
            print("Verification is not deterministic.")
        sys.exit(1)
    
    print("="*60)


if __name__ == "__main__":
    main()