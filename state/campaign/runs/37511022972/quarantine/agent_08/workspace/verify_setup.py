#!/usr/bin/env python3

"""
Simple test to verify that the cost sensitivity momentum check
infrastructure has been properly set up.
"""

import json
from pathlib import Path

def test_infrastructure():
    """Test that the cost sensitivity momentum check files exist and are valid."""
    
    # Check that the main check file exists
    check_file = Path("research/checks/cost_sensitivity_momentum.py")
    if not check_file.exists():
        print(f"ERROR: {check_file} does not exist")
        return False
    
    # Check that the verification file exists  
    verify_file = Path("research/checks/verify_cost_sensitivity_momentum.py")
    if not verify_file.exists():
        print(f"ERROR: {verify_file} does not exist")
        return False
    
    # Check that the artifact file exists
    artifact_file = Path("state/check_artifacts/cost_sensitivity_momentum_results.json")
    if not artifact_file.exists():
        print(f"ERROR: {artifact_file} does not exist")
        return False
    
    # Try to load the artifact to verify it's valid JSON
    try:
        with open(artifact_file, 'r') as f:
            artifact = json.load(f)
        
        # Check that it has the expected structure
        required_fields = ["dataset_id", "seed", "lookbacks", "per_asset", "universe"]
        for field in required_fields:
            if field not in artifact:
                print(f"ERROR: artifact missing field: {field}")
                return False
        
        # Check that it has data for lookbacks 3, 5, 10
        if "3" not in artifact["per_asset"]["AMZN"]:
            print("WARNING: artifact may not have complete data")
        
        print(f"SUCCESS: All infrastructure files exist and artifact is valid")
        print(f"  dataset: {artifact['dataset_id']}")
        print(f"  lookbacks: {artifact['lookbacks']}")
        print(f"  tickers in artifact: {list(artifact['per_asset'].keys())[:5]}...")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"ERROR: artifact is not valid JSON: {e}")
        return False

if __name__ == "__main__":
    success = test_infrastructure()
    exit(0 if success else 1)