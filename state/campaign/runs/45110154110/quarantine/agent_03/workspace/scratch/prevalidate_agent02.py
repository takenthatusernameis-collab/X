"""Read-only pre-validation: replicate validate_campaign_agent's touched/
protected/unexpected-campaign-path computation for agent 2."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".github/scripts")
import validate_campaign_agent as v

RUN_ID = "45110154110"
AGENT = 2
agent_dir = Path("state/campaign/runs") / RUN_ID / "agents"
before = json.load(open(agent_dir / f"agent_{AGENT:02d}_before.json"))
before_hashes = before.get("hashes", {})

after_paths = v.current_changed()
touched = [p for p in after_paths if before_hashes.get(p) != v.sha(Path(p))]
protected = sorted(
    p for p in after_paths
    if p in v.PROTECTED_FILES or any(p.startswith(pre) for pre in v.PROTECTED_PREFIXES)
)
expected_record = f"state/campaign/runs/{RUN_ID}/agents/agent_{AGENT:02d}.json"
allowed_prefix = f"state/campaign/runs/{RUN_ID}/agents/agent_{AGENT:02d}_"
unexpected = [
    p for p in touched
    if p.startswith("state/campaign/") and p != expected_record and not p.startswith(allowed_prefix)
]

print("TOUCHED (my changes):")
for p in touched:
    print("  ", p)
print("PROTECTED (must be empty):", protected)
print("UNEXPECTED_CAMPAIGN_PATHS (must be empty):", unexpected)

python_paths = [p for p in touched if p.endswith(".py") and Path(p).exists()]
print("PYTHON_PATHS_TO_COMPILE:", python_paths)
print("REGRESSION_NEEDED:", any(p.startswith(("research/", "tests/", "examples/")) for p in touched))