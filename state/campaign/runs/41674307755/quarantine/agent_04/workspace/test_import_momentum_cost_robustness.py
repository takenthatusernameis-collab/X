#!/usr/bin/env python3
"""Test import of momentum_cost_robustness module."""
import sys
sys.path.insert(0, '.')
try:
    import research.checks.momentum_cost_robustness
    print("SUCCESS: momentum_cost_robustness module imports successfully")
except Exception as e:
    print(f"ERROR importing momentum_cost_robustness: {e}")
    import traceback
    traceback.print_exc()