import sys
sys.path.insert(0, '/home/runner/work/X/X')

try:
    import research
    print(f"research imported successfully from: {research.__file__}")
    
    import research.backtest as bt
    print(f"research.backtest imported successfully from: {bt.__file__}")
    
    import research.data.preflight
    print(f"research.data.preflight imported successfully")
    
    print("All imports successful!")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)