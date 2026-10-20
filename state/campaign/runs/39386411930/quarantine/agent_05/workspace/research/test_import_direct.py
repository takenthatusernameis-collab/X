import sys
sys.path.insert(0, '.')

try:
    import research.backtest as bt
    print("Import successful!")
    print(f"backtest module: {bt}")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()