import json

p = "state/campaign/runs/45056319162/agents/agent_01.json"
d = json.load(open(p))

required = [
    "agent_number", "role", "task_id", "objective", "bottleneck", "question",
    "action", "changed", "verified", "unverified", "observed_effect",
    "uncertainty_targeted", "uncertainty_reduced", "process_decision",
    "research_result", "decision", "next", "candidate_tasks",
    "complexity_added", "failure_class", "task_selection_observation",
    "campaign_slot", "global_agent_number",
]
missing = [k for k in required if k not in d]
assert not missing, f"MISSING KEYS: {missing}"

canonical_decision = ['NO_SUBSTANTIVE_ACTION', 'REJECT', 'RETAIN', 'UNVERIFIED', 'USEFUL_CHANGE', 'VERIFIED_NEGATIVE_RESULT']
canonical_process = ['IMPROVE', 'REJECT', 'RETAIN', 'UNVERIFIED']
canonical_tso = ['IMPROVED', 'UNCHANGED', 'UNVERIFIED', 'WORSENED']
canonical_fc = ['LIVENESS', 'NONE', 'PERSISTENCE', 'RESEARCH_EXECUTION', 'SCOPE_VIOLATION', 'TOOL_FAILURE', 'TRANSIENT_GATEWAY', 'UNKNOWN', 'VALIDATION_FAILURE']

assert d["decision"] in canonical_decision, d["decision"]
assert d["process_decision"] in canonical_process, d["process_decision"]
assert d["task_selection_observation"] in canonical_tso, d["task_selection_observation"]
assert d["failure_class"] in canonical_fc, d["failure_class"]

assert isinstance(d["changed"], list)
assert isinstance(d["verified"], list)
assert isinstance(d["unverified"], list)
assert isinstance(d["candidate_tasks"], list)

assert (d["agent_number"], d["campaign_slot"], d["global_agent_number"]) == (1, 1, 1)

if d["failure_class"] == "NONE":
    pass

# next must contain exactly one bounded next action: enforce it is a single string sentence
assert isinstance(d["next"], str) and len(d["next"]) > 0

print("VALID: agent_01.json conforms to schema + canonical enums")
print("decision=", d["decision"], "| process_decision=", d["process_decision"],
      "| failure_class=", d["failure_class"], "| task_selection_observation=", d["task_selection_observation"])
print("changed=", len(d["changed"]), "verified=", len(d["verified"]),
      "unverified=", len(d["unverified"]), "candidate_tasks=", len(d["candidate_tasks"]))