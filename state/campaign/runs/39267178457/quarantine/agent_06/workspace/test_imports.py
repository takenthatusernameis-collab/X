#!/usr/bin/env python3
"""Simple test to verify the momentum_cost module can be imported."""

import sys
sys.path.insert(0, '.')

try:
    import research.backtest
    print("✓ research.backtest imports successfully")
    
    # Check if the key functions exist
    assert hasattr(research.backtest, 'momentum_signals'), "momentum_signals not found"
    print("✓ momentum_signals function found")
    
    # Try to import the cost check
    from research.checks import momentum_cost
    print("✓ research.checks.momentum_cost imports successfully")
    
    # Check if the key function exists
    assert hasattr(momentum_cost, 'main'), "main function not found in momentum_cost"
    print("✓ main function found in momentum_cost")
    
    # Check the BacktestConfig import
    from research.backtest import BacktestConfig
    print("✓ BacktestConfig imports successfully")
    
    print("\nAll imports successful! The momentum_cost module is ready to run.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except AssertionError as e:
    print(f"✗ Assertion error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)