#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path.cwd()
if str(project_root) != "/home/runner/work/X/X":
    sys.path.insert(0, str(project_root))

sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

try:
    import research.backtest as bt
    print("SUCCESS: Imported research.backtest")
    print(f"Version: {bt.__version__}")
    
    # Now try to import and run the cost sensitivity module
    from research.checks.cost_sensitivity_momentum import main
    print("SUCCESS: Imported cost_sensitivity_momentum")
    print("Now running main()...")
    
    # Run main function
    result = main()
    print(f"Main returned: {result}")
    
except ImportError as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)