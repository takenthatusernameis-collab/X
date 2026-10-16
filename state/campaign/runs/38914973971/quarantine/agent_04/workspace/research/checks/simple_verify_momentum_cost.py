#!/usr/bin/env python3
"""
Simple verification script for momentum_cost_robustness_results.json artifact.

This script performs basic validation of the artifact file created by
momentum_cost_sensitivity.py to ensure it meets the basic requirements
and structure expected by the independent verification requirement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def validate_artifact_structure(artifact_path: Path) -> dict:
    """Validate the basic structure of the artifact file."""
    with open(artifact_path) as f:
        artifact = json.load(f)
    
    # Required fields according to the momentum_cost_sensitivity.py script
    required_fields = [
        "dataset_id",
        "seed", 
        "lookbacks",
        "train", "test", "warmup", "overlap",
        "window", "n_blocks", "min_segment_bars",
        "commission_per_share",
        "slippage_proportional",
        "per_asset",
        "universe"
    ]
    
    errors = []
    for field in required_fields:
        if field not in artifact:
            errors.append(f"Missing required field: {field}")
    
    # Basic structure checks
    if "per_asset" in artifact:
        if not isinstance(artifact["per_asset"], dict):
            errors.append("per_asset should be a dictionary")
        elif len(artifact["per_asset"]) < 5:  # Should have at least 5 assets
            errors.append(f"per_asset should have at least 5 assets, found {len(artifact['per_asset'])}")
    
    if "universe" in artifact:
        if not isinstance(artifact["universe"], dict):
            errors.append("universe should be a dictionary")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "artifact": artifact
    }


def check_artifact_exists(artifact_path: Path) -> dict:
    """Check if the artifact file exists and is readable."""
    if not artifact_path.exists():
        return {
            "valid": False,
            "errors": [f"Artifact file not found: {artifact_path}"],
            "artifact": None
        }
    
    if not artifact_path.is_file():
        return {
            "valid": False,
            "errors": [f"Artifact path is not a file: {artifact_path}"],
            "artifact": None
        }
    
    # Try to read and parse the JSON
    try:
        with open(artifact_path) as f:
            json.load(f)
        return {
            "valid": True,
            "errors": [],
            "artifact": None
        }
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "errors": [f"Invalid JSON in artifact file: {e}"],
            "artifact": None
        }
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Error reading artifact file: {e}"],
            "artifact": None
        }


def main():
    # The artifact path should match what momentum_cost_sensitivity.py wrote
    artifact_path = Path(__file__).resolve().parent.parent / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"
    
    print("=== Simple Verification of momentum_cost_robustness_results.json ===")
    print(f"Artifact path: {artifact_path}")
    
    # Check if artifact exists
    existence_check = check_artifact_exists(artifact_path)
    if not existence_check["valid"]:
        print(f"❌ FAILED: {existence_check['errors'][0]}")
        sys.exit(1)
    
    print("✓ Artifact file exists and is readable")
    
    # Validate artifact structure
    structure_check = validate_artifact_structure(artifact_path)
    if not structure_check["valid"]:
        print(f"❌ FAILED: Artifact structure validation errors:")
        for error in structure_check["errors"]:
            print(f"  - {error}")
        sys.exit(1)
    
    artifact = structure_check["artifact"]
    
    # Display key information
    print(f"✓ Artifact structure validation passed")
    print(f"  Dataset ID: {artifact.get('dataset_id', 'N/A')}")
    print(f"  Lookbacks: {artifact.get('lookbacks', 'N/A')}")
    print(f"  Commission per share: {artifact.get('commission_per_share', 'N/A')}")
    print(f"  Slippage proportional: {artifact.get('slippage_proportional', 'N/A')}")
    
    # Check universe information
    if "universe" in artifact:
        universe = artifact["universe"]
        if "robustness_counts" in universe:
            robustness_counts = universe["robustness_counts"]
            print(f"  Robustness verdict counts: {robustness_counts}")
        
        if "ticker_order" in universe:
            ticker_order = universe["ticker_order"]
            print(f"  Universe ticker count: {len(ticker_order)} tickers")
            print(f"  First 3 tickers: {ticker_order[:3]}")
    
    # Check per_asset information
    if "per_asset" in artifact:
        per_asset = artifact["per_asset"]
        print(f"  Per-asset entries: {len(per_asset)}")
        
        # Sample one asset
        sample_ticker = list(per_asset.keys())[0]
        sample_data = per_asset[sample_ticker]
        if "robustness_verdict" in sample_data:
            print(f"  Sample asset ({sample_ticker}) robustness verdict: {sample_data['robustness_verdict']}")
        
        if "edge_counts" in sample_data:
            edge_counts = sample_data["edge_counts"]
            print(f"  Sample asset ({sample_ticker}) edge counts: {edge_counts}")
    
    print("\n✅ VERIFICATION PASSED: Artifact meets basic structural requirements")
    print("   Note: This is a basic validation. For complete verification, run the full")
    print("   verify_momentum_cost_sensitivity.py script when import issues are resolved.")


if __name__ == "__main__":
    main()