import sys
import os
sys.path.insert(0, '/home/runner/work/X/X')

print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path[0]}")

try:
    import research
    print(f"research imported from: {research.__file__}")
    
    import research.backtest as bt
    print(f"research.backtest imported from: {bt.__file__}")
    
    print("Import successful!")
    
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)