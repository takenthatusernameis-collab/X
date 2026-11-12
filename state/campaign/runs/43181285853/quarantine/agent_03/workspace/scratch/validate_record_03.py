import json
import sys

sys.path.insert(0, ".github/scripts")
import campaign_controller as controller
import validate_campaign_agent as validator

record = json.loads(open("state/campaign/runs/43181285853/agents/agent_03.json").read())
contract = json.loads(open("state/campaign/runs/43181285853/agents/agent_03_contract.json").read())

failures = validator.validate_record(record, 3, contract)
print("record_failures:", failures)

for task in record["candidate_tasks"]:
    try:
        controller.validate_task(task)
        print("candidate", task["task_id"], "OK")
    except SystemExit as exc:
        print("candidate", task["task_id"], "INVALID:", exc)

print("decision:", record["decision"], "process_decision:", record["process_decision"],
      "obs:", record["task_selection_observation"], "failure_class:", record["failure_class"])
print("next_lines:", len([x for x in record["next"].splitlines() if x.strip()]))
sys.exit(1 if failures else 0)