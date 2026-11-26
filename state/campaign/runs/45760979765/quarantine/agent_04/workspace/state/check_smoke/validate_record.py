import json
import sys

rec_path = "state/campaign/runs/45760979765/agents/agent_02.json"
art_path = "state/check_artifacts/momentum_cost_sensitivity_results.json"

with open(rec_path) as f:
    rec = json.load(f)
required = ["agent_number", "role", "task_id", "objective", "bottleneck", "question",
            "action", "changed", "verified", "unverified", "observed_effect",
            "uncertainty_targeted", "uncertainty_reduced", "process_decision",
            "research_result", "decision", "next", "candidate_tasks",
            "complexity_added", "failure_class", "task_selection_observation",
            "campaign_slot", "global_agent_number"]
missing = [k for k in required if k not in rec]
print("record missing fields:", missing)
assert not missing
assert rec["agent_number"] == 2 and rec["campaign_slot"] == 2 and rec["global_agent_number"] == 2
assert isinstance(rec["changed"], list) and isinstance(rec["verified"], list) and isinstance(rec["unverified"], list)
assert rec["decision"] in ['NO_SUBSTANTIVE_ACTION', 'REJECT', 'RETAIN', 'UNVERIFIED', 'USEFUL_CHANGE', 'VERIFIED_NEGATIVE_RESULT']
assert rec["process_decision"] in ['IMPROVE', 'REJECT', 'RETAIN', 'UNVERIFIED']
assert rec["task_selection_observation"] in ['IMPROVED', 'UNCHANGED', 'UNVERIFIED', 'WORSENED']
assert rec["failure_class"] in ['LIVENESS', 'NONE', 'PERSISTENCE', 'RESEARCH_EXECUTION', 'SCOPE_VIOLATION', 'TOOL_FAILURE', 'TRANSIENT_GATEWAY', 'UNKNOWN', 'VALIDATION_FAILURE']
# next must be exactly one bounded action (single string, no list)
assert isinstance(rec["next"], str) and rec["next"].strip()
print("record OK; decision=", rec["decision"], "process=", rec["process_decision"],
      "selection=", rec["task_selection_observation"], "failure=", rec["failure_class"])

with open(art_path) as f:
    art = json.load(f)
print("artifact OK; universe class conservative=",
      art["universe"]["classification_conservative"],
      "realistic=", art["universe"]["classification_realistic"],
      "robust_assets=", art["universe"]["zero_cost_robust_assets"])
print("VALIDATION PASSED")