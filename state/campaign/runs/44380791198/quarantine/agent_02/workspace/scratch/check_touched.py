import json
import sys
from pathlib import Path

sys.path.insert(0, ".github/scripts")
import validate_campaign_agent as v

run_id = "44380791198"
agent = 1
before = json.load(open(f"state/campaign/runs/{run_id}/agents/agent_{agent:02d}_before.json"))
before_hashes = before.get("hashes", {})
after_paths = v.current_changed()
touched = [p for p in after_paths if before_hashes.get(p) != v.sha(Path(p))]
protected = sorted(
    p for p in after_paths
    if p in v.PROTECTED_FILES or any(p.startswith(x) for x in v.PROTECTED_PREFIXES)
)
expected_record = f"state/campaign/runs/{run_id}/agents/agent_{agent:02d}.json"
unexpected = [
    p for p in touched
    if p.startswith("state/campaign/")
    and p not in {expected_record}
    and not p.startswith(f"state/campaign/runs/{run_id}/agents/agent_{agent:02d}_")
]
print("after_paths:")
for p in after_paths:
    print("  ", p)
print("touched:", touched)
print("protected:", protected)
print("unexpected_campaign_paths:", unexpected)
print("substantive_changed:", [p for p in touched if not p.startswith("state/campaign/") and p != "state/worker_progress.md"])