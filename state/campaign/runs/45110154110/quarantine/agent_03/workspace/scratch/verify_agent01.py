import json
import sys

sys.path.insert(0, ".github/scripts")
import campaign_controller as cc
import validate_campaign_agent as v

rec = json.load(open("state/campaign/runs/45110154110/agents/agent_01.json"))
contract = json.load(open("state/campaign/runs/45110154110/agents/agent_01_contract.json"))

print("VALIDATE_RECORD_FAILURES:", v.validate_record(rec, 1, contract))
print("NEXT_SINGLE_NONEMPTY_LINE:", len([x for x in rec["next"].splitlines() if x.strip()]) == 1)
for i, t in enumerate(rec["candidate_tasks"]):
    try:
        cc.validate_task(t)
        print("CANDIDATE_%d_VALID:" % i, t["task_id"])
    except SystemExit as e:
        print("CANDIDATE_%d_INVALID:" % i, e)
print("ENUMS_OK:", rec["decision"] in cc.DECISIONS,
      rec["process_decision"] in cc.PROCESS_DECISIONS,
      rec["task_selection_observation"] in cc.SELECTION_OBSERVATIONS,
      rec["failure_class"] in cc.FAILURE_CLASSES)
print("IDENTITY:", rec["agent_number"], rec["campaign_slot"], rec["global_agent_number"], rec["task_id"], rec["role"])