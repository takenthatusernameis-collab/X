"""Debug parameter grid and baseline format."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

# Test the correct format
print("Testing parameter grid_around...")
baseline = (("lookback", 5.0),)
grid = bt.parameter_grid_around(baseline)
print(f"Baseline: {baseline}")
print(f"Grid length: {len(grid)}")
print(f"Grid[0]: {grid[0]}")
print(f"dict(baseline): {dict(baseline)}")
print()

# Test with multiple parameters
print("Testing with multiple parameters...")
baseline2 = (("fast", 20.0), ("slow", 60.0))
grid2 = bt.parameter_grid_around(baseline2)
print(f"Baseline: {baseline2}")
print(f"Grid length: {len(grid2)}")
print(f"Grid[0]: {grid2[0]}")
print(f"dict(baseline): {dict(baseline2)}")
