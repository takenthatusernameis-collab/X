import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".github/scripts").resolve()))
import validate_campaign_agent as v

record = json.loads(Path("state/campaign/runs/43776172539/agents/agent_02.json").read_text())
contract = json.loads(Path("state/campaign/runs/43776172539/agents/agent_02_contract.json").read_text())
failures = v.validate_record(record, 2, contract)
print("failures:", failures)
print("task_id contract:", contract.get("task_id"), "role:", contract.get("role"))
print("record task_id:", record.get("task_id"), "role:", record.get("role"))
sys.exit(1 if failures else 0)