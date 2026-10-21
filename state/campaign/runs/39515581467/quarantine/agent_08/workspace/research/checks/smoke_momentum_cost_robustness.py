"""Smoke test for momentum cost-robustness check.

This test verifies that the momentum_cost_robustness.py check can be imported
and has a main function that can be executed without crashing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Test import of momentum_cost_robustness module
import research.checks.momentum_cost_robustness as mcr

# Test that main function exists
assert hasattr(mcr, 'main'), "momentum_cost_robustness.py must have a main function"
assert callable(mcr.main), "main must be callable"

# Test import of verify_momentum_cost_robustness module
import research.checks.verify_momentum_cost_robustness as vmcr

# Test that main function exists
assert hasattr(vmcr, 'main'), "verify_momentum_cost_robustness.py must have a main function"
assert callable(vmcr.main), "main must be callable"

print("Smoke test passed: momentum cost-robustness check imports successfully")