import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import subprocess
import sys
result = subprocess.run([sys.executable, "state/campaign/runs/38189254405/quarantine/agent_05/workspace/research/checks/verify_momentum_lookback_sweep_costs.py"], 
                       capture_output=True, text=True, timeout=300)
print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print("Return code:", result.returncode)
