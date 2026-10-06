#!/usr/bin/env python3

import sys
from pathlib import Path

# Calculate the correct path to add to sys.path
# We're in: /home/runner/work/X/X/research/checks
# We need to add: /home/runner/work/X/X

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
print(f"Script path: {script_path}")
print(f"Project root: {project_root}")

sys.path.insert(0, str(project_root))

try:
    import research.backtest as bt
    print("SUCCESS: Imported research.backtest")
    print(f"Version: {bt.__version__}")
    
    # Now try to run the main function from the cost sensitivity module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cost_sensitivity_momentum", 
        script_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["cost_sensitivity_momentum"] = module
    spec.loader.exec_module(module)
    
    print("SUCCESS: Loaded cost_sensitivity_momentum module")
    print("Now running main()...")
    
    # Run main function
    result = module.main()
    print(f"Main returned: {result}")
    
except ImportError as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)