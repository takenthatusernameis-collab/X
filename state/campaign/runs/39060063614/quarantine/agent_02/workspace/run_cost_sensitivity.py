#!/usr/bin/env python3
"""Launcher for cost sensitivity momentum script."""
import sys
from pathlib import Path

# Add project root to sys.path so we can import research modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Now run the cost sensitivity script directly
exec(open(str(Path(__file__).resolve().parent) + "/cost_sensitivity_momentum.py").read())