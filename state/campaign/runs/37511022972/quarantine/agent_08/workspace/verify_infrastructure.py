#!/usr/bin/env python3

"""
Simple verification test to confirm cost sensitivity momentum check setup.
This test verifies the infrastructure is in place without running the full computations.
"""

import json
from pathlib import Path

def verify_setup():
    """Verify that the cost sensitivity momentum check infrastructure is properly set up."""
    
    print("=== Cost Sensitivity Momentum Check Infrastructure Verification ===")
    
    # Check that the main check file exists
    check_file = Path("research/checks/cost_sensitivity_momentum.py")
    if not check_file.exists():
        print(f"❌ FAIL: {check_file} does not exist")
        return False
    
    print(f"✅ Found {check_file}")
    
    # Check that the verification file exists  
    verify_file = Path("research/checks/verify_cost_sensitivity_momentum.py")
    if not verify_file.exists():
        print(f"❌ FAIL: {verify_file} does not exist")
        return False
    
    print(f"✅ Found {verify_file}")
    
    # Check that the artifact file exists
    artifact_file = Path("state/check_artifacts/cost_sensitivity_momentum_results.json")
    if not artifact_file.exists():
        print(f"❌ FAIL: {artifact_file} does not exist")
        return False
    
    print(f"✅ Found {artifact_file}")
    
    # Try to load the artifact to verify it's valid JSON
    try:
        with open(artifact_file, 'r') as f:
            artifact = json.load(f)
        
        # Check that it has the expected structure
        required_fields = ["dataset_id", "seed", "lookbacks", "per_asset", "universe"]
        missing_fields = [field for field in required_fields if field not in artifact]
        if missing_fields:
            print(f"❌ FAIL: artifact missing fields: {missing_fields}")
            return False
        
        print(f"✅ Artifact has valid structure")
        print(f"   dataset: {artifact['dataset_id']}")
        print(f"   seed: {artifact['seed']}")
        print(f"   lookbacks: {artifact['lookbacks']}")
        print(f"   tickers: {len(artifact['per_asset'])} tickers")
        print(f"   universe verdict counts: {artifact['universe']['robustness_counts']}")
        
        # Check that it has data for lookbacks 3, 5, 10
        amzn_data = artifact['per_asset'].get('AMZN', {})
        for lb in ['3', '5', '10']:
            if lb in amzn_data:
                print(f"✅ AMZN has lookback {lb} data")
            else:
                print(f"⚠️  AMZN missing lookback {lb} data")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ FAIL: artifact is not valid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ FAIL: error checking artifact: {e}")
        return False

if __name__ == "__main__":
    print("Cost Sensitivity Momentum Check Infrastructure Verification")
    print("=" * 60)
    
    success = verify_setup()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SUCCESS: Cost sensitivity momentum check infrastructure is properly set up")
        print("   The files are in place and the artifact is valid")
        print("   Ready to run the actual cost-sensitivity check and verification")
    else:
        print("❌ FAILURE: Infrastructure setup is incomplete or incorrect")
        print("   Please check the missing or invalid files above")
    
    exit(0 if success else 1)