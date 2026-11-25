import json
import sys

REC = "state/campaign/runs/45639223642/agents/agent_02.json"
REQUIRED = [
    "agent_number", "role", "task_id", "objective", "bottleneck", "question",
    "action", "changed", "verified", "unverified", "observed_effect",
    "uncertainty_targeted", "uncertainty_reduced", "process_decision",
    "research_result", "decision", "next", "candidate_tasks",
    "complexity_added", "failure_class", "task_selection_observation",
]
DECISIONS = ["NO_SUBSTANTIVE_ACTION", "REJECT", "RETAIN", "UNVERIFIED",
             "USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT"]
PROC = ["IMPROVE", "REJECT", "RETAIN", "UNVERIFIED"]
TSEL = ["IMPROVED", "UNCHANGED", "UNVERIFIED", "WORSENED"]
FAIL = ["LIVENESS", "NONE", "PERSISTENCE", "RESEARCH_EXECUTION",
        "SCOPE_VIOLATION", "TOOL_FAILURE", "TRANSIENT_GATEWAY", "UNKNOWN",
        "VALIDATION_FAILURE"]
TASK_FIELDS = [
    "task_id", "role", "objective", "primary_question", "scope", "out_of_scope",
    "deliverable", "bottleneck", "evidence_basis", "expected_information_gain",
    "downstream_leverage", "falsification_power", "verifiability", "effort",
    "complexity", "execution_risk", "status", "stop_condition",
    "success_criterion", "verification_requirement",
]

with open(REC) as f:
    d = json.load(f)

errs = []
for k in REQUIRED:
    if k not in d:
        errs.append("missing field: " + k)
if d.get("agent_number") != 2:
    errs.append("agent_number != 2")
if d.get("campaign_slot") != 2:
    errs.append("campaign_slot != 2")
if d.get("global_agent_number") != 2:
    errs.append("global_agent_number != 2")
if d.get("decision") not in DECISIONS:
    errs.append("bad decision: " + str(d.get("decision")))
if d.get("process_decision") not in PROC:
    errs.append("bad process_decision: " + str(d.get("process_decision")))
if d.get("task_selection_observation") not in TSEL:
    errs.append("bad task_selection_observation")
if d.get("failure_class") not in FAIL:
    errs.append("bad failure_class")
for arr in ("changed", "verified", "unverified", "candidate_tasks"):
    if not isinstance(d.get(arr), list):
        errs.append(arr + " is not a list")
# next must be exactly one bounded action (single string, non-empty)
if not isinstance(d.get("next"), str) or not d.get("next").strip():
    errs.append("next must be a non-empty string")
# candidate task schema
for i, t in enumerate(d.get("candidate_tasks", [])):
    for k in TASK_FIELDS:
        if k not in t:
            errs.append("candidate_task_%d missing: %s" % (i, k))

if errs:
    print("INVALID:")
    for e in errs:
        print("  -", e)
    sys.exit(1)
print("VALID: all required fields present; canonical values OK; %d candidate tasks well-formed"
      % len(d["candidate_tasks"]))
print("decision=%s process_decision=%s task_selection=%s failure_class=%s research_result=%s"
      % (d["decision"], d["process_decision"], d["task_selection_observation"],
         d["failure_class"], d["research_result"]))