"""Local schema/canonical-value check for a campaign agent record (scratch helper)."""
import json
import sys

PATH = "state/campaign/runs/43956531905/agents/agent_02.json"

REQUIRED = [
    "agent_number", "role", "task_id", "objective", "bottleneck", "question",
    "action", "changed", "verified", "unverified", "observed_effect",
    "uncertainty_targeted", "uncertainty_reduced", "process_decision",
    "research_result", "decision", "next", "candidate_tasks",
    "complexity_added", "failure_class", "task_selection_observation",
]
DECISIONS = {"NO_SUBSTANTIVE_ACTION", "REJECT", "RETAIN", "UNVERIFIED",
             "USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT"}
PROCESS = {"IMPROVE", "REJECT", "RETAIN", "UNVERIFIED"}
TASKSEL = {"IMPROVED", "UNCHANGED", "UNVERIFIED", "WORSENED"}
FAILURE = {"LIVENESS", "NONE", "PERSISTENCE", "RESEARCH_EXECUTION",
           "SCOPE_VIOLATION", "TOOL_FAILURE", "TRANSIENT_GATEWAY", "UNKNOWN",
           "VALIDATION_FAILURE"}

with open(PATH) as f:
    r = json.load(f)

errs = []
for k in REQUIRED:
    if k not in r:
        errs.append(f"missing field: {k}")
for k in ("changed", "verified", "unverified", "candidate_tasks"):
    if not isinstance(r.get(k), list):
        errs.append(f"{k} is not a list")
if r.get("agent_number") != 2:
    errs.append("agent_number != 2")
if r.get("campaign_slot") != 2:
    errs.append("campaign_slot != 2")
if r.get("global_agent_number") != 2:
    errs.append("global_agent_number != 2")
if r.get("decision") not in DECISIONS:
    errs.append(f"bad decision: {r.get('decision')}")
if r.get("process_decision") not in PROCESS:
    errs.append(f"bad process_decision: {r.get('process_decision')}")
if r.get("task_selection_observation") not in TASKSEL:
    errs.append(f"bad task_selection_observation: {r.get('task_selection_observation')}")
if r.get("failure_class") not in FAILURE:
    errs.append(f"bad failure_class: {r.get('failure_class')}")
if not isinstance(r.get("next"), str) or not r.get("next"):
    errs.append("next must be a non-empty string")
for ct in r.get("candidate_tasks", []):
    for k in ("task_id", "role", "objective", "primary_question", "status"):
        if k not in ct:
            errs.append(f"candidate task {ct.get('task_id')} missing {k}")

if errs:
    print("FAIL")
    for e in errs:
        print(" -", e)
    sys.exit(1)
print("PASS: all required fields present with canonical values")
print("decision:", r["decision"], "| process_decision:", r["process_decision"],
      "| task_selection_observation:", r["task_selection_observation"],
      "| failure_class:", r["failure_class"])
print("candidate_tasks:", [c["task_id"] for c in r["candidate_tasks"]])