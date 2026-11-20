import json, sys
sys.path.insert(0, ".github/scripts")
import campaign_controller as c

rec = json.load(open("state/campaign/runs/44697139653/agents/agent_01.json"))
contract = json.load(open("state/campaign/runs/44697139653/agents/agent_01_contract.json"))

problems = []
for t in rec.get("candidate_tasks", []):
    try:
        c.validate_task(t)
    except SystemExit as exc:
        problems.append(f"candidate_task_invalid: {exc}")

if rec["decision"] not in c.DECISIONS:
    problems.append("invalid_decision")
if rec["process_decision"] not in c.PROCESS_DECISIONS:
    problems.append("invalid_process_decision")
if rec["task_selection_observation"] not in c.SELECTION_OBSERVATIONS:
    problems.append("invalid_task_selection_observation")
if rec["failure_class"] not in c.FAILURE_CLASSES:
    problems.append("invalid_failure_class")
if len([x for x in rec["next"].splitlines() if x.strip()]) != 1:
    problems.append("next_not_one_line")
if not (isinstance(rec["changed"], list) and isinstance(rec["verified"], list) and isinstance(rec["unverified"], list)):
    problems.append("list_fields_invalid")
if not isinstance(rec["candidate_tasks"], list):
    problems.append("candidate_tasks_not_list")

required = [
    "agent_number", "global_agent_number", "campaign_slot", "role", "task_id",
    "objective", "bottleneck", "question", "action", "changed", "verified",
    "unverified", "observed_effect", "uncertainty_targeted", "uncertainty_reduced",
    "process_decision", "research_result", "decision", "next", "candidate_tasks",
    "complexity_added", "failure_class", "task_selection_observation",
]
missing = [k for k in required if k not in rec]
if missing:
    problems.append(f"missing_fields:{missing}")
if rec.get("agent_number") != 1:
    problems.append("agent_number_mismatch")
if rec.get("campaign_slot") != 1:
    problems.append("campaign_slot_mismatch")
if rec.get("global_agent_number") != contract.get("global_agent_number"):
    problems.append("global_agent_number_mismatch")
if rec.get("task_id") != contract.get("task_id"):
    problems.append("task_id_mismatch")
if rec.get("role") != contract.get("role"):
    problems.append("role_mismatch")

print("PROBLEMS:", problems)
print("RESULT:", "PASS" if not problems else "FAIL")