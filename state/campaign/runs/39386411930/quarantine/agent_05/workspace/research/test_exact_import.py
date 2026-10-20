import sys
from pathlib import Path

# Replicate what the script does
script_path = Path("/home/runner/work/X/X/state/campaign/runs/39386411930/quarantine/agent_05/workspace/research/checks/cost_sensitive_momentum.py")
print(f"Script path: {script_path}")

# Add the same path as the script
added_path = script_path.parent.parent.parent / "data"
print(f"Adding to sys.path: {added_path}")
sys.path.insert(0, str(added_path))

# Try the same import as the script
try:
    from preflight import main as preflight_main
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()