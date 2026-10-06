import json
path = "/home/runner/work/X/X/state/campaign/runs/37426042713/agents/agent_01.json"
data = json.load(open(path))
required = ["agent_number","role","task_id","objective","bottleneck","question","action",
            "changed","verified","unverified","observed_effect","uncertainty_targeted",
            "uncertainty_reduced","process_decision","research_result","decision","next",
            "candidate_tasks","complexity_added","failure_class","task_selection_observation"]
for k in required:
    assert k in data, f"missing {k}"
assert isinstance(data["changed"], list)
assert isinstance(data["verified"], list)
assert isinstance(data["unverified"], list)
assert isinstance(data["candidate_tasks"], list)
assert data["process_decision"] in ("IMPROVE","REJECT","RETAIN","UNVERIFIED")
assert data["decision"] in ("NO_SUBSTANTIVE_ACTION","REJECT","RETAIN","UNVERIFIED","USEFUL_CHANGE","VERIFIED_NEGATIVE_RESULT")
assert data["task_selection_observation"] in ("IMPROVED","UNCHANGED","UNVERIFIED","WORSENED")
assert data["failure_class"] == "NONE"
assert data["complexity_added"] == "NONE"
assert isinstance(data["failure_class"], str) and data["failure_class"]
assert len(data["next"]) > 0
print("JSON valid; required fields present;", len(data["verified"]), "verified items;", len(data["candidate_tasks"]), "candidate tasks")
print("decision:", data["decision"]); print("process_decision:", data["process_decision"]); print("task_selection_observation:", data["task_selection_observation"])
