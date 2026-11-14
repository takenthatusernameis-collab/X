import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".github/scripts").resolve()))
import validate_campaign_agent as v
import campaign_controller as controller

rec = json.loads(Path("state/campaign/runs/43638934408/agents/agent_02.json").read_text())
contract = json.loads(Path("state/campaign/runs/43638934408/agents/agent_02_contract.json").read_text())
fails = v.validate_record(rec, 2, contract)
print("record failures:", fails)
for t in rec["candidate_tasks"]:
    try:
        controller.validate_task(t)
        print("task", t["task_id"], "OK")
    except SystemExit as e:
        print("task", t["task_id"], "INVALID:", e)