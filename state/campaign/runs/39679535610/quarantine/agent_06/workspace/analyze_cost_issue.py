"""Analysis of the momentum cost robustness check issue.

This script compares the results from the momentum cost robustness check
with the expected results to understand why there's a mismatch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    print("=== Analysis of Momentum Cost Robustness Check ===\n")
    
    artifact = load_artifact()
    
    print("Published universe-level results:")
    print(f"  robust_assets: {artifact['universe']['robust_assets']}")
    print(f"  sensitive_assets: {artifact['universe']['sensitive_assets']}")
    print(f"  no_edge_assets: {artifact['universe']['no_edge_assets']}")
    print(f"  total_assets: {artifact['universe']['total_assets']}")
    print()
    
    print("Per-asset verdict counts (lookback=5):")
    verdict_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0, 
                      "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    
    for ticker in artifact["per_asset"]:
        verdict = artifact["per_asset"][ticker]["5"]["verdict"]
        verdict_counts[verdict] += 1
        medians = artifact["per_asset"][ticker]["5"]["medians"]
        has_positive_edge = verdict == "REGIME_STABLE" and all(m > 0 for m in medians)
        
        print(f"  {ticker}: {verdict} medians={medians} positive_edge={has_positive_edge}")
    
    print(f"\nTotal verdict counts: {verdict_counts}")
    
    print("\n=== Key Question ===")
    print("The published artifact shows 6 ROBUST assets according to the cost check.")
    print("But the verification script is getting different medians.")
    print("\nPossible causes:")
    print("1. The cost check script has a bug (not applying costs correctly)")
    print("2. The verification script is recomputing incorrectly")
    print("3. There's a difference in the cost application mechanism")
    print("4. The published artifact was generated with a different implementation")
    
    print("\n=== Most Likely Cause ===")
    print("Looking at the cost check script, line 79:")
    print("  actual_config = config if is_null else create_null_config()")
    print("This is backwards! It should be:")
    print("  actual_config = create_null_config() if is_null else config")
    print("\nMore importantly, the 'actual_config' variable is never used.")
    print("The cost check is not actually applying costs!")
    
    print("\n=== Expected vs Actual ===")
    print("Expected (with costs): Lower/negative medians due to commission and slippage")
    print("Actual (artifact): Positive medians (doesn't reflect cost impact)")
    
    print("\n=== Conclusion ===")
    print("The momentum cost robustness check has a fundamental bug:")
    print("1. It passes the wrong config to segment_result")
    print("2. It doesn't actually apply costs in the backtesting")
    print("3. The published artifact is based on flawed methodology")
    print("4. The verification script is showing the correct results")


if __name__ == "__main__":
    main()
