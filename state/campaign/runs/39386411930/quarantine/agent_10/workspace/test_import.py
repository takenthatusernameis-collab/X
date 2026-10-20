import sys
from pathlib import Path

# Test import from repository root
sys.path.insert(0, '/home/runner/work/X/X')

try:
    import research
    print("SUCCESS: research module found at:", research.__file__)
    
    import research.backtest as bt
    print("SUCCESS: research.backtest imported")
    
except ImportError as e:
    print("FAILURE: Cannot import research module")
    print("Error:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)