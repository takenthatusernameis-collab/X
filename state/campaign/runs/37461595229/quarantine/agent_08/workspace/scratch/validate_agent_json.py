import json
p = "/home/runner/work/X/X/state/campaign/runs/37461595229/agents/agent_07.json"
d = json.load(open(p))
required = ["agent_number","campaign_slot","global_agent_number","role","task_id","objective","bottleneck","question","action","changed","verified","unverified","observed_effect","uncertainty_targeted","uncertainty_reduced","process_decision","decision","research_result","next","candidate_tasks","complexity_added","failure_class","task_selection_observation"]
missing = [f for f in required if f not in d]
print("MISSING:", missing)
print("identity:", d["agent_number"], d["campaign_slot"], d["global_agent_number"])
print("values:", d["process_decision"], d["decision"], d["research_result"], d["task_selection_observation"], d["failure_class"], d["complexity_added"])
print("counts:", len(d["changed"]), len(d["verified"]), len(d["unverified"]), len(d["candidate_tasks"]))
print("VALID JSON")
