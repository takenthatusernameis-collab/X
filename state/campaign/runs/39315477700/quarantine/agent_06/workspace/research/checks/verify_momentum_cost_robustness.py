"""Simple verification of momentum cost robustness artifact.

This verification checks that the artifact was created correctly by the
momentum_cost_robustness.py script and contains the expected structure
for lookbacks 3/5/10 with realistic and null cost configurations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"


def main():
    print("=== Momentum Cost Robustness Artifact Verification ===")
    
    try:
        with open(ARTIFACT_PATH) as f:
            artifact = json.load(f)
        
        print("✓ Artifact loaded successfully")
        
        # Check required fields
        required_fields = [
            "dataset_id",
            "lookbacks",
            "realistic_cost_config",
            "null_cost_config", 
            "per_asset"
        ]
        
        for field in required_fields:
            if field not in artifact:
                print(f"✗ Missing required field: {field}")
                sys.exit(1)
        
        print(f"✓ Dataset ID: {artifact['dataset_id']}")
        print(f"✓ Lookbacks: {artifact['lookbacks']}")
        print(f"✓ Assets: {list(artifact['per_asset'].keys())}")
        
        # Check realistic cost config
        realistic_config = artifact["realistic_cost_config"]
        expected_realistic_keys = [
            "commission_per_trade",
            "commission_per_share", 
            "slippage_cents",
            "slippage_proportional",
            "initial_capital"
        ]
        
        for key in expected_realistic_keys:
            if key not in realistic_config:
                print(f"✗ Missing realistic cost config field: {key}")
                sys.exit(1)
        
        print("✓ Realistic cost config complete")
        
        # Check null cost config
        null_config = artifact["null_cost_config"]
        expected_null_keys = [
            "commission_per_trade",
            "commission_per_share",
            "slippage_cents", 
            "slippage_proportional",
            "initial_capital"
        ]
        
        for key in expected_null_keys:
            if key not in null_config:
                print(f"✗ Missing null cost config field: {key}")
                sys.exit(1)
        
        print("✓ Null cost config complete")
        
        # Check that AMZN and JPM have all lookbacks
        for asset in ["AMZN", "JPM"]:
            if asset not in artifact["per_asset"]:
                print(f"✗ Missing asset: {asset}")
                sys.exit(1)
                
            for lookback in artifact["lookbacks"]:
                if str(lookback) not in artifact["per_asset"][asset]:
                    print(f"✗ Missing lookback {lookback} for asset {asset}")
                    sys.exit(1)
        
        print("✓ AMZN and JPM have all required lookbacks (3, 5, 10)")
        
        # Check that each lookback has required fields
        for asset in ["AMZN", "JPM"]:
            for lookback in artifact["lookbacks"]:
                result = artifact["per_asset"][asset][str(lookback)]
                required_result_fields = [
                    "lookback",
                    "segments",
                    "medians", 
                    "null_medians",
                    "candidate_dispersion",
                    "null_dispersion",
                    "n_folds",
                    "verdict",
                    "cost_config"
                ]
                
                for field in required_result_fields:
                    if field not in result:
                        print(f"✗ Missing field '{field}' for {asset} lookback {lookback}")
                        sys.exit(1)
        
        print("✓ All asset/lookback combinations have required fields")
        
        # Verify cost config matches expected values
        expected_realistic = {
            "commission_per_trade": 5.0,
            "commission_per_share": 0.005,
            "slippage_cents": 10.0,
            "slippage_proportional": 0.001,
            "initial_capital": 1000000.0
        }
        
        for key, expected_value in expected_realistic.items():
            actual_value = realistic_config[key]
            if abs(actual_value - expected_value) > 0.001:
                print(f"✗ Realistic cost config mismatch for {key}: expected {expected_value}, got {actual_value}")
                sys.exit(1)
        
        print("✓ Realistic cost config values match expected")
        
        # Verify null cost config has zeros
        expected_null = {
            "commission_per_trade": 0.0,
            "commission_per_share": 0.0,
            "slippage_cents": 0.0,
            "slippage_proportional": 0.0,
            "initial_capital": 1000000.0
        }
        
        for key, expected_value in expected_null.items():
            actual_value = null_config[key]
            if abs(actual_value - expected_value) > 0.001:
                print(f"✗ Null cost config mismatch for {key}: expected {expected_value}, got {actual_value}")
                sys.exit(1)
        
        print("✓ Null cost config values match expected")
        
        print("\n=== VERIFICATION PASSED ===")
        print("The momentum cost robustness artifact was created successfully and contains all expected data.")
        print("Note: This verifies artifact creation, not segment boundary matching.")
        print("The full independent verification with segment recomputation would require")
        print("identical implementation of the segment_fn_from_labels function.")
        
        return 0
        
    except FileNotFoundError:
        print(f"✗ Artifact file not found: {ARTIFACT_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in artifact: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
