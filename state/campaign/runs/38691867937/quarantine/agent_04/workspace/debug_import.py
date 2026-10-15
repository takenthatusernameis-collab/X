import sys
sys.path.insert(0, '.')

print("Testing imports...")
try:
    import research
    print("SUCCESS: research module imported")
    
    print("Attempting to import research.backtest...")
    import research.backtest
    print("SUCCESS: research.backtest imported")
    
    print("Attempting to access research.backtest.momentum_signals...")
    from research.backtest import momentum_signals
    print("SUCCESS: momentum_signals imported")
    
    print("Attempting to access research.backtest.random_signals...")
    from research.backtest import random_signals
    print("SUCCESS: random_signals imported")
    
    print("All imports successful!")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
