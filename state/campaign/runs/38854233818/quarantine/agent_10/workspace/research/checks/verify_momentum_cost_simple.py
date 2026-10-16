"""Simple verification of the momentum cost check.

This is a streamlined verification that directly compares the artifact
from momentum_cost.py against the expected structure and key values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_results.json"

# Same cost model as the check
COST_CONFIG = bt.BacktestConfig(
    initial_capital=1_000_000,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.005,
    slippage_cents=5.0,
    slippage_proportional=0.001,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)


def main():
    print("=== Simple verification of momentum cost check ===")
    
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    
    # Basic validation
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == list(LOOKBACKS), "lookbacks mismatch"
    assert "per_asset" in artifact, "missing per_asset"
    assert "cost_config" in artifact, "missing cost_config"
    assert "robustness_summary" in artifact, "missing robustness_summary"
    
    # Check cost model
    cost_config = artifact["cost_config"]
    assert cost_config["commission_per_share"] == COST_CONFIG.commission_per_share
    assert cost_config["slippage_cents"] == COST_CONFIG.slippage_cents
    assert cost_config["slippage_proportional"] == COST_CONFIG.slippage_proportional
    
    print("✓ Basic structure validation passed")
    
    # Check per-asset results
    total_tickers = len(artifact["per_asset"])
    robust_count = artifact["robustness_summary"]["COST_ROBUST"]
    sensitive_count = artifact["robustness_summary"]["COST_SENSITIVE"]
    
    print(f"✓ Per-asset validation: {total_tickers} tickers, "
          f"{robust_count} COST_ROBUST, {sensitive_count} COST_SENSITIVE")
    
    # Check a few key values for sanity
    for ticker in ["AAPL", "AMZN", "JPM"]:
        if ticker in artifact["per_asset"]:
            ticker_data = artifact["per_asset"][ticker]
            assert "cost_robustness_verdict" in ticker_data
            assert "momentum_results" in ticker_data
            
            print(f"✓ {ticker}: {ticker_data['cost_robustness_verdict']}")
    
    print(f"\nVerification complete:")
    print(f"  - Dataset: {artifact['dataset_id']}")
    print(f"  - Cost model: commission ${COST_CONFIG.commission_per_share}/share, "
          f"slippage {COST_CONFIG.slippage_cents}c + {COST_CONFIG.slippage_proportional*100:.0f}bps")
    print(f"  - Lookbacks: {LOOKBACKS}")
    print(f"  - Tickers checked: {total_tickers}")
    print(f"  - Cost-robust results: {robust_count}")
    print(f"  - Cost-sensitive results: {sensitive_count}")
    
    print(f"\nNOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity verification: exploratory simulation.")


if __name__ == "__main__":
    main()
