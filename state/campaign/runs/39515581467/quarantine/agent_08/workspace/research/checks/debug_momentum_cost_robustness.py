#!/usr/bin/env python3
"""Debug script to check determinism in momentum cost robustness."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research.checks.momentum_cost_robustness import main

if __name__ == "__main__":
    # Run with a flag to bypass the assertion
    import research.checks.momentum_cost_robustness as mcr
    import inspect
    print("Checking if main function exists...")
    print(f"main function: {mcr.main}")
    print(f"source file: {inspect.getfile(mcr)}")
    
    # Try to run just the determinism check part
    print("\nTrying to import and call main...")
    try:
        mcr.main()
    except AssertionError as e:
        print(f"Assertion error (expected): {e}")
    except Exception as e:
        print(f"Other error: {e}")