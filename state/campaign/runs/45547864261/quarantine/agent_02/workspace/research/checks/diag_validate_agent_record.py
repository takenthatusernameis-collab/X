import json
from pathlib import Path

p = Path("state/campaign/runs/45547864261/agents/agent_02.json")
rec = json.loads(p.read_text())
required = ["agent_number", "role", "task_id", "objective", "bottleneck",
            "question", "action", "changed", "verified", "unverified",
            "observed_effect", "uncertainty_targeted", "uncertainty_reduced",
            "process_decision", "research_result", "decision", "next",
            "candidate_tasks", "complexity_added", "failure_class",
            "task_selection_observation", "campaign_slot", "global_agent_number"]
missing = [k for k in required if k not in rec]
print("missing:", missing)
print("agent_number:", rec["agent_number"], "campaign_slot:", rec["campaign_slot"], "global:", rec["global_agent_number"])
print("decision:", rec["decision"], "| process_decision:", rec["process_decision"], "| failure_class:", rec["failure_class"], "| task_selection:", rec["task_selection_observation"])
print("changed is list:", isinstance(rec["changed"], list), "verified is list:", isinstance(rec["verified"], list), "unverified is list:", isinstance(rec["unverified"], list))
print("candidate_tasks count:", len(rec["candidate_tasks"]))
for ct in rec["candidate_tasks"]:
    ct_req = ["status", "expected_information_gain", "downstream_leverage", "falsification_power", "verifiability", "effort", "complexity", "execution_risk"]
    cmiss = [k for k in ct_req if k not in ct]
    print("  candidate", ct.get("task_id"), "missing:", cmiss)
print("VALID JSON, all required fields present" if not missing else "INVALID")