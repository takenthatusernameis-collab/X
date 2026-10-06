import json, sys
p = '/home/runner/work/X/X/state/campaign/runs/37452390298/agents/agent_04.json'
try:
    d = json.load(open(p))
    print("JSON VALID")
    print("record fields:", sorted(d.keys()))
    required = ["agent_number","role","task_id","objective","bottleneck","question",
                "action","changed","verified","unverified","observed_effect",
                "uncertainty_targeted","uncertainty_reduced","process_decision",
                "research_result","decision","next","candidate_tasks",
                "complexity_added","failure_class","task_selection_observation"]
    missing = [f for f in required if f not in d]
    print("missing required:", missing if missing else "none")
    print("agent_number:", d.get("agent_number"), "| campaign_slot:", d.get("campaign_slot"),
          "| global_agent_number:", d.get("global_agent_number"))
    print("decision:", d.get("decision"), "| process_decision:", d.get("process_decision"),
          "| task_selection_observation:", d.get("task_selection_observation"))
    print("verified count:", len(d.get("verified", [])), "| unverified count:", len(d.get("unverified", [])))
    print("changed count:", len(d.get("changed", [])))
    print("candidate_tasks:", len(d.get("candidate_tasks", [])))
except Exception as e:
    print("JSON ERROR:", e)
    sys.exit(1)
