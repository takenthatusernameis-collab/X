#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

print(f"Added to path: {str(Path(__file__).resolve().parent.parent.parent)}")
print(f"Script location: {Path(__file__).resolve()}")

try:
    import research.backtest as bt
    print("SUCCESS: Imported research.backtest")
    print(f"Version: {bt.__version__}")
    
    # Now try to import the cost sensitivity module
    from research.checks.cost_sensitivity_momentum import main
    print("SUCCESS: Imported cost_sensitivity_momentum")
    
except ImportError as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)