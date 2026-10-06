#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import research.backtest as bt
    print("SUCCESS: Imported research.backtest")
    print(f"Version: {bt.__version__}")
except ImportError as e:
    print(f"FAILED: {e}")
    sys.exit(1)