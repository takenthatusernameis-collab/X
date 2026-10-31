#!/usr/bin/env python3
import sys
from pathlib import Path

# Add the current directory to the path (where research/ is located)
sys.path.insert(0, str(Path(__file__).resolve().parent))

print("Testing import...")

try:
    import research.backtest
    print("SUCCESS: Imported research.backtest")
    
    from research.backtest.engine import BacktestConfig, Signal, walk_forward
    print("SUCCESS: Imported BacktestConfig, Signal, walk_forward")
    
    from research.backtest.real_data import load_universe
    print("SUCCESS: Imported load_universe")
    
    from research.backtest.regime_stability import momentum_signals
    print("SUCCESS: Imported momentum_signals")
    
    print("\nAll imports successful!")
    
except ImportError as e:
    print(f"FAILED: Import error: {e}")
    print(f"sys.path: {sys.path}")
    import os
    print(f"Current directory: {os.getcwd()}")
    
    # Check what modules are available
    print("\nChecking available modules...")
    try:
        import importlib
        spec = importlib.util.find_spec("research")
        print(f"research module spec: {spec}")
        
        if spec:
            print(f"research location: {spec.origin}")
    except Exception as e2:
        print(f"Error checking module: {e2}")