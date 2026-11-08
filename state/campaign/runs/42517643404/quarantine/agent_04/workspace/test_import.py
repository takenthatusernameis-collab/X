import sys
from pathlib import Path

# Try the same path setup as momentum_lookback_sweep.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
print(f"Added to sys.path: {Path(__file__).resolve().parent.parent.parent}")

try:
    import research.backtest as bt
    print("Successfully imported research.backtest")
except Exception as e:
    print(f"Failed to import research.backtest: {e}")
    print(f"Current sys.path[0]: {sys.path[0] if sys.path else 'empty'}")
