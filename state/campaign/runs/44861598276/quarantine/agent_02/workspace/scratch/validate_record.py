import json
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / ".github" / "scripts"))
import validate_campaign_agent as v  # noqa: E402

run = "44861598276"
agent = 2
contract = json.loads((root / "state" / "campaign" / "runs" / run / "agents" / f"agent_{agent:02d}_contract.json").read_text())
record = json.loads((root / "state" / "campaign" / "runs" / run / "agents" / f"agent_{agent:02d}.json").read_text())
failures = v.validate_record(record, agent, contract)
print("failures:", failures)
print("next_lines:", len([x for x in record["next"].splitlines() if x.strip()]))
print("candidate_task_count:", len(record["candidate_tasks"]))
sys.exit(1 if failures else 0)