from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
print("Current directory:", Path.cwd())
print("Looking for research module...")
import research
print("research module found")
print("research.__file__:", research.__file__)
print("research.backtest:", research.backtest)