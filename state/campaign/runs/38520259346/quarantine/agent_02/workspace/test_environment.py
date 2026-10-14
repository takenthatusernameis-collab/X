"""Minimal test to verify the environment."""
from __future__ import annotations

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import numpy as np
    print(f"NumPy imported successfully: {np.__version__}")
    
    import research.backtest as bt
    print(f"Research backtest imported successfully: {bt.__version__}")
    
    print("All imports successful - environment is ready")
    
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)