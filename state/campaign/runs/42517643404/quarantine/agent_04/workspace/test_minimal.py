from pathlib import Path
import sys

print("Current working directory:", Path.cwd())
print("Script location:", Path(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
print("Added to sys.path:", sys.path[0])

print("\nTrying to import research.backtest...")
try:
    import research.backtest as bt
    print("Successfully imported research.backtest")
    print("Backtest config:", bt.BacktestConfig)
except Exception as e:
    print(f"Failed to import research.backtest: {e}")
    import traceback
    traceback.print_exc()
