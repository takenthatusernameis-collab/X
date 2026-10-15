#!/usr/bin/env python3
import sys
import os

# Add the repository root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the momentum cost robustness test
from research.checks.momentum_cost_robustness import main
sys.exit(main())