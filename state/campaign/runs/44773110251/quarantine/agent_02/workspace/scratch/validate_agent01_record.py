import json
import sys

sys.path.insert(0, ".github/scripts")
import validate_campaign_agent as v  # noqa: E402

record = json.load(open("state/campaign/runs/44773110251/agents/agent_01.json"))
contract = json.load(open("state/campaign/runs/44773110251/agents/agent_01_contract.json"))
failures = v.validate_record(record, 1, contract)
print("FAILURES:", failures)
print("STATUS:", "VALID" if not failures else "INVALID")