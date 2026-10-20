import sys
from pathlib import Path

# Replicate what the script does
script_path = Path("/home/runner/work/X/X/state/campaign/runs/39386411930/quarantine/agent_05/workspace/research/checks/cost_sensitive_momentum.py")
print(f"Script path: {script_path}")
print(f"Script parent: {script_path.parent}")
print(f"Script parent.parent: {script_path.parent.parent}")
print(f"Script parent.parent.parent: {script_path.parent.parent.parent}")

sys.path.insert(0, str(script_path.parent.parent.parent))
print(f"Added to sys.path: {sys.path[0]}")

# List what's in the path
import os
print(f"\nContents of {sys.path[0]}:")
for item in os.listdir(sys.path[0]):
    item_path = os.path.join(sys.path[0], item)
    if os.path.isdir(item_path):
        print(f"  DIR: {item}")
    else:
        print(f"  FILE: {item}")

# Check if research module exists
research_path = os.path.join(sys.path[0], "research")
print(f"\nChecking if research exists at: {research_path}")
if os.path.exists(research_path):
    print(f"  research exists as: {os.path.abspath(research_path)}")
    if os.path.exists(os.path.join(research_path, "__init__.py")):
        print(f"  research has __init__.py")
    else:
        print(f"  research does NOT have __init__.py")
        
    # Check if backtest exists
    backtest_path = os.path.join(research_path, "backtest")
    if os.path.exists(backtest_path):
        print(f"  research.backtest exists")
    else:
        print(f"  research.backtest does NOT exist")
else:
    print(f"  research does NOT exist")

# Now try the actual import
try:
    import research.backtest as bt
    print(f"\nImport successful! backtest module: {bt}")
except Exception as e:
    print(f"\nImport failed: {e}")
    import traceback
    traceback.print_exc()