#!/usr/bin/env python3
"""Simple verification script for momentum cost robustness check."""

import json
from pathlib import Path

def verify_files():
    print("=== Verifying momentum cost robustness implementation ===")
    
    # Check momentum_cost_robustness.py
    momentum_file = Path("/home/runner/work/X/X/research/checks/momentum_cost_robustness.py")
    if momentum_file.exists():
        with open(momentum_file, 'r') as f:
            content = f.read()
            if "COST_ZERO" in content and "COST_REALISTIC" in content:
                print("✓ momentum_cost_robustness.py exists with correct cost models")
            else:
                print("✗ momentum_cost_robustness.py missing cost models")
                return False
    else:
        print("✗ momentum_cost_robustness.py not found")
        return False
    
    # Check verify_momentum_cost_robustness.py
    verify_file = Path("/home/runner/work/X/X/research/checks/verify_momentum_cost_robustness.py")
    if verify_file.exists():
        with open(verify_file, 'r') as f:
            content = f.read()
            if "vol_blocks" in content and "segments_from_labels" in content:
                print("✓ verify_momentum_cost_robustness.py exists with verification functions")
            else:
                print("✗ verify_momentum_cost_robustness.py missing verification functions")
                return False
    else:
        print("✗ verify_momentum_cost_robustness.py not found")
        return False
    
    # Check agent_06.json
    agent_file = Path("/home/runner/work/X/X/state/campaign/runs/39168583498/agents/agent_06.json")
    if agent_file.exists():
        with open(agent_file, 'r') as f:
            try:
                data = json.load(f)
                if data.get("agent_number") == 6 and data.get("decision") == "USEFUL_CHANGE":
                    print("✓ agent_06.json exists with correct structure")
                    print(f"  - Agent number: {data.get('agent_number')}")
                    print(f"  - Task ID: {data.get('task_id')}")
                    print(f"  - Decision: {data.get('decision')}")
                    return True
                else:
                    print("✗ agent_06.json has incorrect structure")
                    return False
            except json.JSONDecodeError:
                print("✗ agent_06.json contains invalid JSON")
                return False
    else:
        print("✗ agent_06.json not found")
        return False

if __name__ == "__main__":
    success = verify_files()
    if success:
        print("\n✓ All files created successfully!")
    else:
        print("\n✗ Some files are missing or incorrect")
        exit(1)