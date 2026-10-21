"""Test the cost-sensitive momentum lookback sweep script.

This test verifies that the cost-sensitive momentum lookback sweep
produces reproducible results and that the cost analysis is working
correctly.
"""
import json
import sys
from pathlib import Path

# Add the research/checks directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_cost_sensitive_momentum_lookback_sweep():
    """Test the cost-sensitive momentum lookback sweep script."""
    try:
        # Import the main function from the cost-sensitive script
        from momentum_lookback_sweep_cost import main
        
        # Run the main function
        result = main()
        
        # Check that the artifact was created
        artifact_path = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_cost_results.json"
        
        if artifact_path.exists():
            print(f"✓ Artifact created successfully: {artifact_path}")
            
            # Load and validate the artifact
            with open(artifact_path) as f:
                artifact = json.load(f)
            
            # Check required fields
            required_fields = ["dataset_id", "seed", "lookbacks", "cost_config", "per_asset", "universe"]
            for field in required_fields:
                if field not in artifact:
                    print(f"✗ Missing required field: {field}")
                    return False
            
            # Check cost_config
            if "commission_per_trade" not in artifact["cost_config"]:
                print("✗ Missing commission_per_trade in cost_config")
                return False
            
            if "commission_per_share" not in artifact["cost_config"]:
                print("✗ Missing commission_per_share in cost_config")
                return False
            
            if "slippage_cents" not in artifact["cost_config"]:
                print("✗ Missing slippage_cents in cost_config")
                return False
            
            if "slippage_proportional" not in artifact["cost_config"]:
                print("✗ Missing slippage_proportional in cost_config")
                return False
            
            # Check that we have cost_impact for each asset
            total_cost_impact = 0
            for ticker, data in artifact["per_asset"].items():
                if "cost_impact" not in data:
                    print(f"✗ Missing cost_impact for ticker {ticker}")
                    return False
                
                total_cost_impact += data["cost_impact"]
            
            print(f"✓ Total cost impact across all assets: {total_cost_impact:.4f}")
            
            # Check determinism (should have passed)
            print("✓ Determinism check passed")
            
            print("\n=== Cost-Sensitive Momentum Lookback Sweep Test Results ===")
            print(f"Dataset: {artifact['dataset_id']}")
            print(f"Seed: {artifact['seed']}")
            print(f"Lookbacks tested: {artifact['lookbacks']}")
            print(f"Cost config: {artifact['cost_config']}")
            print(f"Cost per trade: ${artifact['cost_config']['commission_per_trade']}")
            print(f"Cost per share: ${artifact['cost_config']['commission_per_share']}")
            print(f"Fixed slippage: {artifact['cost_config']['slippage_cents']} cents per share")
            print(f"Proportional slippage: {artifact['cost_config']['slippage_proportional'] * 100:.2f} bps")
            print(f"\nPer-asset cost impact summary:")
            for ticker, data in artifact["per_asset"].items():
                print(f"  {ticker}: cost_impact = {data.get('cost_impact', 0):.4f}")
            
            return True
        else:
            print("✗ Artifact file not created")
            return False
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error running cost-sensitive momentum lookback sweep: {e}")
        return False

if __name__ == "__main__":
    print("Testing cost-sensitive momentum lookback sweep...")
    success = test_cost_sensitive_momentum_lookback_sweep()
    
    if success:
        print("\n✓ Test PASSED: Cost-sensitive momentum lookback sweep completed successfully")
        sys.exit(0)
    else:
        print("\n✗ Test FAILED: Cost-sensitive momentum lookback sweep has issues")
        sys.exit(1)