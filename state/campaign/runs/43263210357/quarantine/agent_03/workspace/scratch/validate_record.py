import json
from pathlib import Path

rec = json.load(open("state/campaign/runs/43263210357/agents/agent_02.json"))
req = ["agent_number", "role", "task_id", "objective", "bottleneck", "question",
       "action", "changed", "verified", "unverified", "observed_effect",
       "uncertainty_targeted", "uncertainty_reduced", "process_decision",
       "research_result", "decision", "next", "candidate_tasks",
       "complexity_added", "failure_class", "task_selection_observation"]
missing = [k for k in req if k not in rec]
print("missing required fields:", missing)
print("decision:", rec["decision"], "| process_decision:", rec["process_decision"],
      "| task_selection_observation:", rec["task_selection_observation"],
      "| failure_class:", rec["failure_class"])
print("changed/verified/unverified are lists:",
      all(isinstance(rec[k], list) for k in ("changed", "verified", "unverified")))
print("n candidate_tasks:", len(rec["candidate_tasks"]))

lines = Path("state/LEARNING_STATE.md").read_text().split("\n")
for i in range(92, 97):
    print(i + 1, repr(lines[i][:60]), "..." if len(lines[i]) > 60 else "")

art = json.load(open("state/check_artifacts/momentum_cost_sensitivity_results.json"))
print("artifact top-level keys:", sorted(art.keys())[:8], "...")
print("artifact cells variants:", list(art["cells"].keys()),
      "lookbacks:", list(art["cells"]["existing"].keys()))