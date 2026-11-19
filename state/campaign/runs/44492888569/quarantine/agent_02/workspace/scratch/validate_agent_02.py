"""Validate the agent_02 durable record and its candidate task schema."""
import importlib.util
import json

spec = importlib.util.spec_from_file_location(
    "campaign_controller", ".github/scripts/campaign_controller.py"
)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

with open("state/campaign/runs/44492888569/agents/agent_02.json") as fh:
    record = json.load(fh)

required = [
    "agent_number", "role", "task_id", "objective", "bottleneck", "question",
    "action", "changed", "verified", "unverified", "observed_effect",
    "uncertainty_targeted", "uncertainty_reduced", "process_decision",
    "research_result", "decision", "next", "candidate_tasks",
    "complexity_added", "failure_class", "task_selection_observation",
]
missing = [k for k in required if k not in record]
print("missing fields:", missing)
assert not missing

for key in ("changed", "verified", "unverified", "candidate_tasks"):
    assert isinstance(record[key], list), key
assert record["decision"] in cc.DECISIONS, record["decision"]
assert record["process_decision"] in cc.PROCESS_DECISIONS, record["process_decision"]
assert record["task_selection_observation"] in cc.SELECTION_OBSERVATIONS
assert record["failure_class"] in cc.FAILURE_CLASSES
assert record["agent_number"] == 2
assert record["campaign_slot"] == 2
assert record["global_agent_number"] == 2

for task in record["candidate_tasks"]:
    cc.validate_task(task)
    print("candidate task valid:", task["task_id"], task["role"], task["status"])

print("agent_02 record valid")