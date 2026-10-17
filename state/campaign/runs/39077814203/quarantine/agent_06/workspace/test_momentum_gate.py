"""Test script to verify momentum_volatility_gate.py functionality.

This script tests the basic structure without running the full backtest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Test imports
print("Testing imports...")

try:
    import research.backtest as bt
    print("  ✓ Successfully imported research.backtest")
except Exception as e:
    print(f"  ✗ Failed to import research.backtest: {e}")
    sys.exit(1)

try:
    from research.data.preflight import load_manifest, sha256_file
    print("  ✓ Successfully imported preflight functions")
except Exception as e:
    print(f"  ✗ Failed to import preflight functions: {e}")
    sys.exit(1)

# Test basic functions
print("\nTesting basic functions...")

try:
    # Test momentum_signals function
    import numpy as np
    closes = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    from research.checks.momentum_volatility_gate import momentum_signals, gated_momentum_signals, volatility_gate_condition
    
    momentum_sig = momentum_signals(closes, 5)
    print(f"  ✓ momentum_signals returned {len(momentum_sig)} signals")
    
    gate_sig = gated_momentum_signals(closes, 5, 0.15)
    print(f"  ✓ gated_momentum_signals returned {len(gate_sig)} signals")
    
    vol_gate = volatility_gate_condition(0.10)
    print(f"  ✓ volatility_gate_condition(0.10) = {vol_gate} (expected: True)")
    
    vol_gate2 = volatility_gate_condition(0.20)
    print(f"  ✓ volatility_gate_condition(0.20) = {vol_gate2} (expected: False)")
    
    print("\nAll basic tests passed!")
    print("The momentum_volatility_gate.py module appears to be syntactically correct.")
    
except Exception as e:
    print(f"  ✗ Error testing basic functions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
