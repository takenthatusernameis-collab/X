#!/usr/bin/env python3
"""Simple test runner for momentum cost-robustness check."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the momentum cost robustness check
from research.checks.momentum_cost_robustness import main

if __name__ == "__main__":
    sys.exit(main())