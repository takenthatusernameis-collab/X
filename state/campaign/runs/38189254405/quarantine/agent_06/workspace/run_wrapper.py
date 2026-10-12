import sys
from pathlib import Path

# Add the repository root to Python path
sys.path.insert(0, str(Path.cwd()))

# Now we can import the script from the quarantine workspace
import importlib.util
spec = importlib.util.spec_from_file_location(
    "verify_costs", 
    "state/campaign/runs/38189254405/quarantine/agent_05/workspace/research/checks/verify_momentum_lookback_sweep_costs.py"
)
module = importlib.util.module_from_spec(spec)

# Execute the module
print("Setting up Python path...")
print("Current working directory:", Path.cwd())
print("Script path:", spec.origin)

spec.loader.exec_module(module)
