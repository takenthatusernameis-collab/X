import json
import sys

path = sys.argv[1]
with open(path) as f:
    rec = json.load(f)

required = [
    "agent_number", "role", "task_id", "objective", "bottleneck", "question",
    "action", "changed", "verified", "unverified", "observed_effect",
    "uncertainty_targeted", "uncertainty_reduced", "process_decision",
    "research_result", "decision", "next", "candidate_tasks",
    "complexity_added", "failure_class", "task_selection_observation",
]
missing = [k for k in required if k not in rec]
print("VALID JSON; missing fields:", missing)

canonical_decision = {"NO_SUBSTANTIVE_ACTION", "REJECT", "RETAIN", "UNVERIFIED", "USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT"}
canonical_process = {"IMPROVE", "REJECT", "RETAIN", "UNVERIFIED"}
canonical_tso = {"IMPROVED", "UNCHANGED", "UNVERIFIED", "WORSENED"}
canonical_fc = {"LIVENESS", "NONE", "PERSISTENCE", "RESEARCH_EXECUTION", "SCOPE_VIOLATION", "TOOL_FAILURE", "TRANSIENT_GATEWAY", "UNKNOWN", "VALIDATION_FAILURE"}

print("decision in canonical:", rec["decision"] in canonical_decision, "->", rec["decision"])
print("process_decision in canonical:", rec["process_decision"] in canonical_process, "->", rec["process_decision"])
print("task_selection_observation in canonical:", rec["task_selection_observation"] in canonical_tso, "->", rec["task_selection_observation"])
print("failure_class in canonical:", rec["failure_class"] in canonical_fc, "->", rec["failure_class"])
print("identity:", rec["agent_number"], rec.get("campaign_slot"), rec.get("global_agent_number"))
print("changed is list:", isinstance(rec["changed"], list), "| verified is list:", isinstance(rec["verified"], list), "| unverified is list:", isinstance(rec["unverified"], list))
print("candidate_tasks count:", len(rec["candidate_tasks"]))
for t in rec["candidate_tasks"]:
    print("  candidate:", t["task_id"], t["role"], t["status"])
print("next:", rec["next"][:120], "...")