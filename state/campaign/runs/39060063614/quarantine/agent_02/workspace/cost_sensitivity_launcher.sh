#!/bin/bash
# Launcher script for cost sensitivity momentum script

cd /home/runner/work/X/X

# Set PYTHONPATH to include the project root
export PYTHONPATH=/home/runner/work/X/X:$PYTHONPATH

# Run the cost sensitivity script from the workspace
python3 state/campaign/runs/39060063614/quarantine/agent_02/workspace/cost_sensitivity_momentum.py