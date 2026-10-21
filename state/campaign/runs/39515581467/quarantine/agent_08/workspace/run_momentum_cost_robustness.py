#!/usr/bin/env python3
"""Minimal wrapper to run momentum cost-robustness check with proper sys.path."""
import sys
from pathlib import Path

# Add the current directory and research directory to Python path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "research"))

# Run the momentum cost robustness check
from state.campaign.runs.39515581467.quarantine.agent_06.workspace.research.checks import momentum_cost_robustness

if __name__ == "__main__":
    sys.exit(momentum_cost_robustness.main())