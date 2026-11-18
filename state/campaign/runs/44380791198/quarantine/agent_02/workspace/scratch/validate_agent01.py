import json
import sys

sys.path.insert(0, ".github/scripts")
import campaign_controller as c
import validate_campaign_agent as v

rec = json.load(open("state/campaign/runs/44380791198/agents/agent_01.json"))
con = json.load(open("state/campaign/runs/44380791198/agents/agent_01_contract.json"))
failures = v.validate_record(rec, 1, con)
print("validate_record failures:", failures)

expected = {
    "agent_number": 1,
    "campaign_slot": 1,
    "global_agent_number": con["global_agent_number"],
}
print("identity expected:", expected)
print("record identity:", {k: rec.get(k) for k in expected})
print("decision valid:", rec["decision"] in c.DECISIONS)
print("process_decision valid:", rec["process_decision"] in c.PROCESS_DECISIONS)
print("task_selection_observation valid:", rec["task_selection_observation"] in c.SELECTION_OBSERVATIONS)
print("failure_class valid:", rec["failure_class"] in c.FAILURE_CLASSES)
print("next single line:", len([x for x in rec["next"].splitlines() if x.strip()]) == 1)
print("candidate_tasks count:", len(rec["candidate_tasks"]))