import json
p = "state/campaign/runs/44226957913/agents/agent_03.json"
d = json.load(open(p))
req = ["agent_number","role","task_id","objective","bottleneck","question","action","changed","verified","unverified","observed_effect","uncertainty_targeted","uncertainty_reduced","process_decision","research_result","decision","next","candidate_tasks","complexity_added","failure_class","task_selection_observation"]
print("OK", d["task_id"], d["decision"], d["process_decision"], d["task_selection_observation"], d["failure_class"])
print("missing", [k for k in req if k not in d])
print("arrays_ok", all(isinstance(d[k], list) for k in ("changed","verified","unverified","candidate_tasks")))
print("identity", d["agent_number"], d.get("campaign_slot"), d.get("global_agent_number"))