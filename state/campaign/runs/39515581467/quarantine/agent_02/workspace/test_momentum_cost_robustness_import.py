#!/usr/bin/env python3
"""Quick test to verify the momentum cost-robustness check files can be imported."""

import sys
from pathlib import Path

# Add the repo root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    # Test import of the main check
    print("Testing import of momentum_cost_robustness...")
    import research.checks.momentum_cost_robustness as mcr
    print("✓ momentum_cost_robustness imported successfully")
    
    # Test import of the verification file
    print("Testing import of verify_momentum_cost_robustness...")
    import research.checks.verify_momentum_cost_robustness as vmcr
    print("✓ verify_momentum_cost_robustness imported successfully")
    
    # Test import of the smoke test
    print("Testing import of smoke_momentum_cost_robustness...")
    import research.checks.smoke_momentum_cost_robustness as sm
    print("✓ smoke_momentum_cost_robustness imported successfully")
    
    print("\nAll imports successful! The momentum cost-robustness check files are ready.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)