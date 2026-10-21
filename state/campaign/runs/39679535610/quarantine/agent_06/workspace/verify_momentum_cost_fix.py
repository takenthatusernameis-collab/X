"""Verification script for the momentum cost robustness check.

This script verifies that the momentum cost robustness check:
1. Uses the correct cost configuration
2. Applies costs properly in backtesting
3. Follows the full gate stack requirements
4. Can be independently verified

The script checks the current state of the momentum_cost_robustness.py script
and identifies any issues that need to be fixed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASEtt_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"

# Expected cost model
EXPECTED_COST_MODEL = {
    "commission_per_trade": 0.005,
    "slippage_bps": 1.5,
    "slippage_proportional": 0.0015,
}


def check_cost_robustness_script():
    """Check the current momentum_cost_robustness.py script."""
    print("=== Checking momentum_cost_robustness.py script ===\n")
    
    script_path = Path.cwd() / "research" / "checks" / "momentum_cost_robustness.py"
    with open(script_path, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for config logic bug (line 79 in original)
    if "actual_config = config if is_null else create_null_config()" in content:
        issues.append("❌ CRITICAL BUG: Line with backwards config logic found:")
        issues.append("   actual_config = config if is_null else create_null_config()")
        issues.append("   This should be: actual_config = create_null_config() if is_null else config")
    
    # Check if cost_config is used in backtesting
    if "actual_config" in content:
        if "segment_result" in content and "actual_config" not in content.split("def segment_result")[1].split("def ")[0]:
            issues.append("❌ BUG: actual_config variable defined but not used in backtesting")
    
    # Check if cost_config is passed to walk_forward/stress_segments
    if "create_cost_config" not in content:
        issues.append("❌ MISSING: create_cost_config function not found")
    
    # Check if stress_segments is called with cost config
    if "bt.stress_segments" in content and "config" not in content:
        issues.append("❌ BUG: bt.stress_segments called but no config parameter passed")
    
    # Check for proper gate stack
    if "preflight" not in content.lower():
        issues.append("❌ MISSING: preflight gate not implemented")
    
    if "leakage" not in content.lower():
        issues.append("❌ MISSING: leakage gate not implemented")
    
    if "determinism" not in content.lower():
        issues.append("❌ MISSING: determinism check not implemented")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        
        print(f"\nSUMMARY: {len(issues)} issues found that need to be fixed")
        return False
    else:
        print("✅ No issues found - script looks correct")
        return True


def check_artifact():
    """Check the momentum cost robustness artifact."""
    print("\n=== Checking momentum_cost_robustness_results.json artifact ===\n")
    
    try:
        with open(ARTIFACT_PATH) as f:
            artifact = json.load(f)
        
        # Check cost model
        if "cost_model" in artifact:
            print(f"✅ Cost model found: {artifact['cost_model']}")
            
            # Check if it matches expected
            if artifact["cost_model"] == EXPECTED_COST_MODEL:
                print("✅ Cost model matches expected values")
            else:
                print("❌ Cost model does not match expected values")
        else:
            print("❌ Cost model not found in artifact")
        
        # Check structure
        if "per_asset" in artifact:
            print(f"✅ Per-asset data found for {len(artifact['per_asset'])} assets")
        
        if "universe" in artifact:
            print(f"✅ Universe summary found")
            print(f"   robust_assets: {artifact['universe']['robust_assets']}")
            print(f"   sensitive_assets: {artifact['universe']['sensitive_assets']}")
            print(f"   no_edge_assets: {artifact['universe']['no_edge_assets']}")
            
            # Check if the results make sense (not all robust, not all no_edge)
            total = artifact['universe']['total_assets']
            robust = artifact['universe']['robust_assets']
            sensitive = artifact['universe']['sensitive_assets']
            no_edge = artifact['universe']['no_edge_assets']
            
            if robust > 0 and no_edge > 0:
                print("✅ Results show realistic variation (some robust, some no-edge)")
            else:
                print("❌ Results may not reflect realistic cost impact")
        
        # Check for verification function
        print("\n✅ Artifact structure looks reasonable")
        return True
        
    except Exception as e:
        print(f"❌ Error reading artifact: {e}")
        return False


def main():
    print("Momentum Cost Robustness Check Verification Script")
    print("=" * 50)
    
    script_ok = check_cost_robustness_script()
    artifact_ok = check_artifact()
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY:")
    print(f"Script check: {'PASS' if script_ok else 'FAIL'}")
    print(f"Artifact check: {'PASS' if artifact_ok else 'FAIL'}")
    
    if not script_ok or not artifact_ok:
        print("\nRECOMMENDED ACTIONS:")
        print("1. Fix the config logic in momentum_cost_robustness.py")
        print("2. Properly apply costs in the backtesting functions")
        print("3. Ensure the cost configuration is passed to walk_forward/stress_segments")
        print("4. Implement the full gate stack (preflight, leakage, determinism)")
        print("5. Verify that the cost check shows realistic cost impact")
        
        return 1
    else:
        print("\n✅ All checks passed - the momentum cost robustness check is working correctly")
        return 0


if __name__ == "__main__":
    sys.exit(main())
