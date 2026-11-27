import json
from pathlib import Path

ROOT = Path("/home/runner/work/X/X")
rec = json.loads((ROOT / "state/campaign/runs/45962051330/agents/agent_02.json").read_text())
con = json.loads((ROOT / "state/campaign/runs/45962051330/agents/agent_02_contract.json").read_text())

DECISIONS = {"USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT", "RETAIN", "REJECT", "UNVERIFIED", "NO_SUBSTANTIVE_ACTION"}
PROCESS = {"IMPROVE", "RETAIN", "REJECT", "UNVERIFIED"}
OBS = {"IMPROVED", "UNCHANGED", "WORSENED", "UNVERIFIED"}
FAIL = {"NONE", "TRANSIENT_GATEWAY", "TOOL_FAILURE", "RESEARCH_EXECUTION", "PERSISTENCE", "LIVENESS", "SCOPE_VIOLATION", "VALIDATION_FAILURE", "UNKNOWN"}
REQ = ["agent_number", "global_agent_number", "campaign_slot", "role", "task_id", "objective", "bottleneck", "question",
       "action", "changed", "verified", "unverified", "observed_effect", "uncertainty_targeted", "uncertainty_reduced",
       "process_decision", "research_result", "decision", "next", "candidate_tasks", "complexity_added", "failure_class",
       "task_selection_observation"]
TASK_REQ = ["task_id", "role", "status", "primary_question", "bottleneck", "objective", "scope", "out_of_scope",
            "deliverable", "success_criterion", "stop_condition", "verification_requirement", "evidence_basis",
            "expected_information_gain", "downstream_leverage", "falsification_power", "verifiability", "effort",
            "complexity", "execution_risk", "tests_preceding_process"]
STATUSES = {"OPEN", "SELECTED", "RESOLVED", "REJECTED", "DEFERRED"}
SEV = {"HIGH", "MEDIUM", "LOW"}

fails = []
for f in REQ:
    if f not in rec:
        fails.append("missing:" + f)
if rec["agent_number"] != 2: fails.append("agent_number")
if rec.get("campaign_slot") != 2: fails.append("campaign_slot")
if rec.get("global_agent_number") != con.get("global_agent_number"): fails.append("global_agent_number")
if rec["task_id"] != con["task_id"]: fails.append("task_id")
if rec["role"] != con["role"]: fails.append("role")
if rec["decision"] not in DECISIONS: fails.append("decision")
if rec["process_decision"] not in PROCESS: fails.append("process_decision")
if rec["task_selection_observation"] not in OBS: fails.append("obs")
if rec["failure_class"] not in FAIL: fails.append("failure_class")
for k in ("changed", "verified", "unverified"):
    if not isinstance(rec[k], list): fails.append("list:" + k)
if not isinstance(rec["candidate_tasks"], list):
    fails.append("candidate_tasks_not_list")
else:
    for i, t in enumerate(rec["candidate_tasks"]):
        for f in TASK_REQ:
            if f not in t: fails.append(f"cand{i}_missing:{f}")
        if t.get("status") not in STATUSES: fails.append(f"cand{i}_status")
        q = str(t.get("primary_question", ""))
        if not q.strip() or len(q) > 500 or q.count("?") > 1: fails.append(f"cand{i}_question")
        for f in TASK_REQ[4:14]:
            if not isinstance(t.get(f), str) or not t.get(f).strip(): fails.append(f"cand{i}_text:{f}")
        for f in ("expected_information_gain", "downstream_leverage", "falsification_power", "verifiability", "effort", "complexity", "execution_risk"):
            if t.get(f) not in SEV: fails.append(f"cand{i}_sev:{f}")
        if not isinstance(t.get("tests_preceding_process"), bool): fails.append(f"cand{i}_bool")
n = rec["next"]
if not isinstance(n, str) or not n.strip(): fails.append("next_missing")
if len([x for x in n.splitlines() if x.strip()]) != 1: fails.append("next_not_single_line")

print("FAILS:", fails if fails else "NONE")
print("decision:", rec["decision"], "| process:", rec["process_decision"], "| obs:", rec["task_selection_observation"], "| failure:", rec["failure_class"])
print("next:", n)