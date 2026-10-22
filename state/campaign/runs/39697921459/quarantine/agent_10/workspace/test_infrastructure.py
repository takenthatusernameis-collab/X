"""Simplified test to verify momentum cost-sensitivity functionality.

This script tests the core momentum signal generation and cost-sensitivity
analysis without running the full walk-forward validation.
"""
import sys
import json
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path.cwd()))

def test_momentum_signals():
    """Test momentum signal generation."""
    try:
        # Try to import the research.backtest module
        import research.backtest as bt
        print("✓ Successfully imported research.backtest")

        # Test momentum signals function
        if hasattr(bt, 'momentum_signals'):
            print("✓ momentum_signals function exists")
            # Test with sample data
            closes = [100.0, 101.0, 102.0, 103.0, 104.0] * 100  # 500 data points
            signals = bt.momentum_signals(closes, lookback=5)
            print(f"✓ Generated {len(signals)} momentum signals")
            
            # Check signal structure
            if signals and len(signals) > 0:
                print(f"✓ First signal: date={signals[0].date}, weight={signals[0].weight}")
                return True
            else:
                print("✗ No signals generated")
                return False
        else:
            print("✗ momentum_signals function not found in research.backtest")
            return False
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_backtest_config():
    """Test BacktestConfig cost model."""
    try:
        import research.backtest as bt
        print("✓ Successfully imported research.backtest")
        
        # Test realistic cost configuration
        realistic = bt.BacktestConfig(
            initial_capital=1e6,
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        )
        print(f"✓ Realistic config: commission_per_trade={realistic.commission_per_trade}")
        
        # Test zero cost configuration
        zero = bt.BacktestConfig(
            initial_capital=1e6,
            commission_per_trade=0.0,
            commission_per_share=0.0,
            slippage_cents=0.0,
            slippage_proportional=0.0,
        )
        print(f"✓ Zero cost config: commission_per_trade={zero.commission_per_trade}")
        
        return True
        
    except Exception as e:
        print(f"✗ BacktestConfig test error: {e}")
        return False

def check_artifact_exists():
    """Check if momentum results artifact exists."""
    artifact_path = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"
    if artifact_path.exists():
        print(f"✓ Momentum results artifact exists: {artifact_path}")
        
        # Try to read and parse it
        try:
            with open(artifact_path, 'r') as f:
                data = json.load(f)
            print(f"✓ Successfully parsed artifact: dataset_id={data.get('dataset_id', 'unknown')}")
            return True
        except Exception as e:
            print(f"✗ Error parsing artifact: {e}")
            return False
    else:
        print(f"✗ Momentum results artifact not found: {artifact_path}")
        return False

def main():
    """Run all tests."""
    print("=== Testing momentum cost-sensitivity infrastructure ===\n")
    
    tests = [
        ("Momentum signals", test_momentum_signals),
        ("BacktestConfig cost models", test_backtest_config),
        ("Existing artifact", check_artifact_exists),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        success = test_func()
        results.append((test_name, success))
        print()
    
    print("=== Test Summary ===")
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! The infrastructure is ready for cost-sensitivity analysis.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the infrastructure.")
        return 1

if __name__ == "__main__":
    exit(main())