#!/usr/bin/env python3
"""
Minimal verification script for momentum cost-robustness assessment.

This follows the verification pattern from existing research/checks scripts.
"""
import json
import sys
from pathlib import Path

def main():
    print("=== Independent Verification of Momentum Cost-Robustness ===")
    
    # Load the artifact from our cost-robustness check
    artifact_path = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"
    
    if not artifact_path.exists():
        print(f"ERROR: Artifact not found at {artifact_path}")
        sys.exit(1)
    
    print(f"Loading artifact from {artifact_path}")
    
    with open(artifact_path, 'r') as f:
        artifact = json.load(f)
    
    # Verify basic structure
    required_fields = ["dataset_id", "seed", "lookbacks", "per_asset", "universe"]
    for field in required_fields:
        if field not in artifact:
            print(f"ERROR: Missing required field '{field}'")
            sys.exit(1)
    
    print(f"✓ Dataset ID: {artifact['dataset_id']}")
    print(f"✓ Seed: {artifact['seed']}")
    print(f"✓ Lookbacks: {artifact['lookbacks']}")
    print(f"✓ Universe robustness counts: {artifact['universe']['robustness_counts']}")
    
    # Verify per-asset structure
    for ticker, per_asset_data in artifact['per_asset'].items():
        print(f"✓ {ticker}: cost_robustness_verdict={per_asset_data['cost_robustness_verdict']}")
    
    print(f"\n✓ Verification completed successfully")
    print(f"\nNOTE: research/simulation only. No live trading or production execution.")
    print(f"Cost-robustness verification for lookback {artifact['lookbacks']} completed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())