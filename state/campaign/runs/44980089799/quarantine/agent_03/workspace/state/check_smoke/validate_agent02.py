import json
import sys
from pathlib import Path

ROOT = Path("/home/runner/work/X/X")
sys.path.insert(0, str(ROOT / ".github" / "scripts"))
import validate_campaign_agent as v  # noqa: E402
import campaign_controller as c  # noqa: E402

agent_dir = ROOT / "state/campaign/runs/44980089799/agents"
rec = json.loads((agent_dir / "agent_02.json").read_text())
contract = json.loads((agent_dir / "agent_02_contract.json").read_text())

fails = v.validate_record(rec, 2, contract)
print("validate_record failures:", fails)
for i, t in enumerate(rec["candidate_tasks"]):
    try:
        c.validate_task(t)
        print("candidate_task", i, t["task_id"], "OK")
    except SystemExit as exc:
        print("candidate_task", i, "FAIL", exc)
print("decision in DECISIONS:", rec["decision"] in c.DECISIONS)
print("process_decision in PROCESS_DECISIONS:", rec["process_decision"] in c.PROCESS_DECISIONS)
print("task_selection_observation in OBS:", rec["task_selection_observation"] in c.SELECTION_OBSERVATIONS)
print("failure_class in FAILURE_CLASSES:", rec["failure_class"] in c.FAILURE_CLASSES)