#!/usr/bin/env python3
"""Simple test to verify momentum cost robustness imports."""

import sys
from pathlib import Path

# Add the repository root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Test imports
try:
    from research.checks.momentum_cost_robustness import (
        CFG_ZERO, CFG_REAL, segment_result
    )
    print("✓ Successfully imported momentum_cost_robustness")
    
    # Test BacktestConfig initialization
    print(f"Zero cost config: commission_per_trade={CFG_ZERO.commission_per_trade}, "
          f"commission_per_share={CFG_ZERO.commission_per_share}, "
          f"slippage_cents={CFG_ZERO.slippage_cents}, "
          f"slippage_proportional={CFG_ZERO.slippage_proportional}")
    
    print(f"Real cost config: commission_per_trade={CFG_REAL.commission_per_trade}, "
          f"commission_per_share={CFG_REAL.commission_per_share}, "
          f"slippage_cents={CFG_REAL.slippage_cents}, "
          f"slippage_proportional={CFG_REAL.slippage_proportional}")
    
    # Test BacktestConfig import (momentum_signals is in research.backtest)
    import research.backtest as bt
    import numpy as np
    test_closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0])
    test_signals = bt.momentum_signals(test_closes, lookback=3)
    print(f"✓ momentum_signals test: generated {len(test_signals)} signals")
    
    print("\n✓ All basic tests passed!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test error: {e}")
    sys.exit(1)
