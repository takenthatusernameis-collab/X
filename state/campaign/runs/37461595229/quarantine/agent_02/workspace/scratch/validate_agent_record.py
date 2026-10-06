import json

PATH = "state/campaign/runs/37461595229/agents/agent_01.json"
REQUIRED = [
    "agent_number", "role", "task_id", "objective", "bottleneck", "question",
    "action", "changed", "verified", "unverified", "observed_effect",
    "uncertainty_targeted", "uncertainty_reduced", "process_decision",
    "research_result", "decision", "next", "candidate_tasks",
    "complexity_added", "failure_class", "task_selection_observation",
]
CANONICAL = {
    "process_decision": ["IMPROVE", "REJECT", "RETAIN", "UNVERIFIED"],
    "task_selection_observation": ["IMPROVED", "UNCHANGED", "UNVERIFIED", "WORSENED"],
    "decision": ["NO_SUBSTANTIVE_ACTION", "REJECT", "RETAIN", "UNVERIFIED", "USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT"],
    "failure_class": ["LIVENESS", "NONE", "PERSISTENCE", "RESEARCH_EXECUTION", "SCOPE_VIOLATION", "TOOL_FAILURE", "TRANSIENT_GATEWAY", "UNKNOWN", "VALIDATION_FAILURE"],
}

d = json.load(open(PATH))
missing = [k for k in REQUIRED if k not in d]
assert not missing, "missing required keys: %s" % missing
assert isinstance(d["changed"], list) and d["changed"], "changed must be a non-empty list"
assert isinstance(d["verified"], list) and d["verified"], "verified must be a non-empty list"
assert isinstance(d["unverified"], list), "unverified must be a list"
assert isinstance(d["candidate_tasks"], list), "candidate_tasks must be a list"
assert isinstance(d["next"], str) and d["next"], "next must be a non-empty string"
for k, allowed in CANONICAL.items():
    assert d[k] in allowed, "%s=%s not in %s" % (k, d[k], allowed)
    assert d[k] in allowed, "%s=%s not in %s" % (k, d[k], allowed)
assert d["agent_number"] == 1 and d["campaign_slot"] == 1 and d["global_agent_number"] == 1
assert d["process_decision"] == "IMPROVE"
assert d["decision"] == "RETAIN"
assert d["task_selection_observation"] == "IMPROVED"
assert d["failure_class"] == "NONE"
assert d["complexity_added"] == "NONE"
print("JSON RECORD VALID")
print("  required keys present:", len(REQUIRED))
print("  canonical values OK")
print("  identity metadata OK (agent=%s slot=%s global=%s)" % (d["agent_number"], d["campaign_slot"], d["global_agent_number"]))
print("  changed=%d verified=%d unverified=%d candidates=%d" % (
    len(d["changed"]), len(d["verified"]), len(d["unverified"]), len(d["candidate_tasks"])))
