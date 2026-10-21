#!/usr/bin/env python3
"""Wrapper script to run momentum_cost_robustness directly."""

import subprocess
import sys

def main():
    """Run the momentum_cost_robustness script."""
    # Use subprocess to run the script directly
    result = subprocess.run(
        [sys.executable, '/home/runner/work/X/X/research/checks/momentum_cost_robustness.py'],
        capture_output=True,
        text=True
    )
    
    # Print stdout and stderr
    print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)
    
    # Exit with the same return code
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()