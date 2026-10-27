"""Simple verification that the momentum cost-robustness check completed successfully.

This script checks that:
1. The cost-robustness artifact was created
2. The artifact has the expected structure
3. Basic validation passes
"""
import json
from pathlib import Path

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"

def main():
    print("=== Simple verification of momentum_cost_robustness.py ===")
    
    # Check if artifact exists
    if not ARTIFACT_PATH.exists():
        print(f"ERROR: Artifact not found at {ARTIFACT_PATH}")
        return 1
    
    print(f"✓ Artifact found at {ARTIFACT_PATH}")
    
    # Load and validate artifact
    with open(ARTIFACT_PATH, "r") as f:
        artifact = json.load(f)
    
    # Basic validation
    assert artifact["dataset_id"] == "yf-ohlcv-universe-2009-to-2026-10-03", "Invalid dataset_id"
    assert artifact["lookbacks"] == [3, 5, 10], "Invalid lookbacks"
    assert "zero_cost" in artifact, "Missing zero_cost in artifact"
    assert "conservative_cost" in artifact, "Missing conservative_cost in artifact"
    
    # Check structure
    for cost_regime in ["zero_cost", "conservative_cost"]:
        assert "per_asset" in artifact[cost_regime], f"Missing per_asset in {cost_regime}"
        for ticker in artifact[cost_regime]["per_asset"]:
            asset_data = artifact[cost_regime]["per_asset"][ticker]
            assert "lookback_medians" in asset_data, f"Missing lookback_medians for {ticker} in {cost_regime}"
            assert "lookback_null_medians" in asset_data, f"Missing lookback_null_medians for {ticker} in {cost_regime}"
            assert "verdicts" in asset_data, f"Missing verdicts for {ticker} in {cost_regime}"
            assert "robustness_verdict" in asset_data, f"Missing robustness_verdict for {ticker} in {cost_regime}"
            
            # Check lookbacks
            for lb in ["3", "5", "10"]:
                assert lb in asset_data["lookback_medians"], f"Missing lookback {lb} in lookback_medians for {ticker}"
                assert lb in asset_data["lookback_null_medians"], f"Missing lookback {lb} in lookback_null_medians for {ticker}"
                assert lb in asset_data["verdicts"], f"Missing lookback {lb} in verdicts for {ticker}"
    
    # Print summary
    print("\n=== Verification Summary ===")
    for cost_regime in ["zero_cost", "conservative_cost"]:
        print(f"\n{cost_regime}:")
        for ticker in artifact[cost_regime]["per_asset"]:
            robustness_verdict = artifact[cost_regime]["per_asset"][ticker]["robustness_verdict"]
            print(f"  {ticker}: {robustness_verdict}")
    
    print("\n✓ All validation checks passed!")
    return 0

if __name__ == "__main__":
    exit(main())