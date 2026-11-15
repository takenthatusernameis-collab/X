import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".github/scripts").resolve()))
import validate_campaign_agent as v

record = json.load(open("state/campaign/runs/43687380362/agents/agent_02.json"))
contract = json.load(open("state/campaign/runs/43687380362/agents/agent_02_contract.json"))
failures = v.validate_record(record, 2, contract)
print("record failures:", failures)
for t in record["candidate_tasks"]:
    try:
        v.controller.validate_task(t)
        print("candidate", t["task_id"], "OK")
    except SystemExit as exc:
        print("candidate", t.get("task_id"), "INVALID:", exc)